#include "PIMLayoutTransform.h"

#include "Bounds.h"
#include "Function.h"
#include "FuseGPUThreadLoops.h"
#include "IRMutator.h"
#include "IRVisitor.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Parameter.h"
#include "Scope.h"
#include "InjectHostDevBufferCopies.h"
#include "Simplify.h"
#include "DeviceArgument.h"
#include "Scope.h"

#include <sstream>

namespace Halide {
namespace Internal {

using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::vector;
using std::set;
using std::pair;
using std::reverse;

// This is superset of ExtractBounds
class ResourceAllocationBuilder : public IRVisitor {
using IRVisitor::visit;
public:
    class ExtractContiguousVariable : public IRVisitor {
    public:
        vector<string> contiguous_variables;
        vector<string> all_variables;
        bool bypass = false;

        using IRVisitor::visit;

        void visit(const Variable* op) override {
            if (!bypass) {
                contiguous_variables.push_back(op->name);
            }
            all_variables.push_back(op->name);
        }

        void visit(const Mul* op) override {
            bypass = true;
            IRVisitor::visit(op);
            bypass = false;
        }
    };

    class BaseContiguousVariable : public IRMutator {
        public: 
        string contiguous_variable;
        bool bypass = false;

        BaseContiguousVariable(string contiguous_variable) {
            this->contiguous_variable = contiguous_variable;
        }
        using IRMutator::visit;

        Expr visit(const Variable* op) override {
            if (!bypass && op->name == contiguous_variable) {
                return 0;
            }
            return IRMutator::visit(op);
        }

        Expr visit(const Mul* op) override {
            bypass = true;
            Expr ret = IRMutator::visit(op);
            bypass = false;
            return ret;
        }
    };

    set<string> shared_memory_buffers;

    Stmt optimized_loop;

    vector<Stmt> stmt_stack;
    vector<pair<Expr, Interval>> bound_stack;
    ResourceAllocationBuilder() { }

    Stmt get_push_stmt(const string &name, const Expr &value, const Expr& size) {
        Expr buffer = Variable::make(type_of<struct halide_buffer_t *>(), name + ".buffer");
        Stmt push_stmt = Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_to", { Variable::make(UInt(32), "dpu_idx"), buffer, value, size }, Call::Extern));
        return push_stmt;
    }

    void visit(const ProducerConsumer* op) override {
        shared_memory_buffers.insert(op->name);
        IRVisitor::visit(op);
    }

    void visit(const Allocate* op) override {
        shared_memory_buffers.insert(op->name);
        IRVisitor::visit(op);
    }

    void add_new_loop(string name, Expr index) {

        ExtractContiguousVariable extractor;
        index.accept(&extractor);
        auto& candidates = extractor.contiguous_variables;
        auto& all_variables = extractor.all_variables;

        ostringstream o;
        o << "add_new_loop: " << name << " ";
        for (auto& c : candidates) {
            o << c << " ";
        }
        debug(2) << o.str() << "\n";

        const For* obsolete_block;
        Expr obsolete_extent = 1;
        for (const auto& block: candidates) {
            for (const auto& stmt: stmt_stack) {
                if (stmt.as<For>() && (stmt.as<For>())->name == block) {
                    obsolete_block = stmt.as<For>();
                    BaseContiguousVariable mutator(obsolete_block->name);
                    index = simplify(Add::make(mutator.mutate(index), obsolete_block->min));
                    obsolete_extent = obsolete_block->extent;
                    break;
                }
            }
            if (obsolete_block) break;
        }

        Stmt push_stmt = get_push_stmt(name, index, obsolete_extent);

        for (vector<Stmt>::reverse_iterator it = stmt_stack.rbegin(); it != stmt_stack.rend(); ++it) {
            if ((*it).as<For>()) {
                const For* f = (*it).as<For>();
                if (f == obsolete_block) continue;
                if (std::find(all_variables.begin(), all_variables.end(), f->name) == all_variables.end()) continue;
                push_stmt = For::make(f->name, f->min, f->extent, f->for_type, f->device_api, push_stmt);
            } else if ((*it).as<IfThenElse>()) {
                const IfThenElse* f = (*it).as<IfThenElse>();
                push_stmt = IfThenElse::make(f->condition, push_stmt, f->else_case); // TODO ; else_case
            } else if ((*it).as<LetStmt>()) {
                const LetStmt* f = (*it).as<LetStmt>();
                push_stmt = LetStmt::make(f->name, f->value, push_stmt);
            } else {
                break;
            }
        }

        if (!optimized_loop.defined())
            optimized_loop = push_stmt;
        else
            optimized_loop = Block::make({ optimized_loop, push_stmt});
    }

    void visit(const Load* op) override {
        if (shared_memory_buffers.find(op->name) == shared_memory_buffers.end())
            add_new_loop(op->name, op->index);
    }

    void visit(const Store* op) override {
        if (shared_memory_buffers.find(op->name) == shared_memory_buffers.end())
            add_new_loop(op->name, op->index);
        IRVisitor::visit(op);
    }

    void visit(const LetStmt* op) override {
        stmt_stack.push_back(op);
        IRVisitor::visit(op);
        // TODO: cond
        stmt_stack.pop_back();
    }

    void visit(const For* op) override {
        stmt_stack.push_back(op);
        IRVisitor::visit(op);
        stmt_stack.pop_back();
    }

    void visit(const IfThenElse* op) override {
        stmt_stack.push_back(op);
        // TODO: cond
        IRVisitor::visit(op);
        stmt_stack.pop_back();
    }

    Stmt get_stmt() {
        return this->optimized_loop;
    }
};

class PIMLayoutTransform : public IRMutator {
public:
    PIMLayoutTransform() {}

    class ThreadLoopMutator : public IRMutator {
        public:
        ThreadLoopMutator(Stmt stmt): stmt(stmt) {}
        Stmt stmt;

        using IRMutator::visit;
        Stmt visit(const For *op) override {
            if (op->for_type == ForType::PIMThread) {
                return For::make(op->name, op->min, op->extent, ForType::Serial, DeviceAPI::None, stmt);
            } else if (op->for_type == ForType::PIMBank) {
                return For::make(op->name, op->min, op->extent, ForType::Serial, DeviceAPI::None, mutate(op->body));
            } else {
                return IRMutator::visit(op);
            }
        }
    };


    class InspectResourceAllocation : public IRVisitor {
    public:
        Expr num_threads[1];
        Expr num_banks[4];

        Expr name_threads[1];
        Expr name_banks[4];

        Expr bank = 0;
        Stmt thread_loop;

        InspectResourceAllocation() {
            for (int i = 0; i < 4; i++) {
                num_banks[i] = 1;
                name_banks[i] = Expr("");
            }
            num_threads[0] = 1;
            name_threads[0] = Expr("");
        }

        using IRVisitor::visit;
        void visit(const For *op) override {
            if (ends_with(op->name, ".__thread_id_x")) {
                num_threads[0] = op->extent;
                thread_loop = For::make(op->name, op->min, op->extent, ForType::Serial, DeviceAPI::None, op->body);
            } else if (ends_with(op->name, ".__bank_id_x")) {
                num_banks[0] = op->extent;
                name_banks[0] = op->name;
                bank = Variable::make(UInt(32), op->name);
            } else if (ends_with(op->name, ".__bank_id_y")) {
                num_banks[1] = op->extent;
                name_banks[1] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), Variable::make(UInt(32), op->name));
            } else if (ends_with(op->name, ".__bank_id_z")) {
                num_banks[2] = op->extent;
                name_banks[2] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), Variable::make(UInt(32), op->name));
            } else if (ends_with(op->name, ".__bank_id_w")) {
                num_banks[3] = op->extent;
                name_banks[3] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), Variable::make(UInt(32), op->name));
            }
            op->body.accept(this);
        }

        Expr get_all_banks() {
            return num_banks[0] * num_banks[1] * num_banks[2] * num_banks[3];
        }

        Expr get_bank() {
            return bank;
        }
    };

private:
    size_t kernel_idx = 0;
    using IRMutator::visit;

    Stmt visit(const For* loop) override {
        if (!ends_with(loop->name, ".__bank_id_x")) return IRMutator::visit(loop);
 
        const string kernel_name = "kernel_" + std::to_string(kernel_idx++); // TODO : to be changed

        InspectResourceAllocation inspect;
        loop->accept(&inspect);

        Stmt stmt_alloc_load = call_extern_and_assert("halide_upmem_alloc_load", { inspect.get_all_banks(), kernel_name });

        Stmt stmt_free = call_extern_and_assert("halide_upmem_free", { kernel_name });

        ResourceAllocationBuilder builder;
        inspect.thread_loop.accept(&builder);
        auto stmt_copy_to = For::make(loop->name+".__allocation", loop->min, loop->extent, ForType::Serial, DeviceAPI::None, LetStmt::make("dpu_idx", inspect.bank, builder.get_stmt()));
        // Stmt for_copy_reduction = For::make(loop->name + ".__reduction", Expr(loop->min), Expr(loop->extent), loop->for_type, DeviceAPI::None, for_copy_allocation);

        
        // I want to duplicate outermost PIM Bank type loop, and remove all other expressions but only remain store/load expressions and loop, if, and let depends to indexing store/load, and convert each store/load into device_copy.

            // TODO : add ALLOC & LOAD
            // TODO : add DISTRIBUTE
            // TODO : add REDUCE
            // TODO : add FREE
        return Block::make({ stmt_alloc_load, stmt_copy_to, IRMutator::visit(loop), stmt_free });
    }
};

Stmt pim_layout_transform(Stmt s) {
    return PIMLayoutTransform().mutate(s);
}

}  // namespace Internal
}  // namespace Halide