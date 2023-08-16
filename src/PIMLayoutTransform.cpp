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

// This is superset of ExtractBounds
class ResourceAllocationBuilder : public IRVisitor {
using IRVisitor::visit;
public:
    vector<Stmt> stmts;
    set<Expr> shared_memory_buffers;
    ResourceAllocationBuilder() {}

    void add_push_stmt(const string &name, const Expr &value) {
        Expr buffer = Variable::make(type_of<struct halide_buffer_t *>(), name + ".buffer");
        Stmt push = Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_to", { Variable::make(UInt(32), "dpu_idx"), buffer, value }, Call::Extern));
        stmts.push_back(push);
    }

    void visit(const ProducerConsumer* op) override {
        shared_memory_buffers.insert(op->name);
        op->body.accept(this);
    }

    void visit(const Load* op) override {
        if (shared_memory_buffers.find(op->name) == shared_memory_buffers.end()) {
            add_push_stmt(op->name, op->index);
        }
    }

    void visit(const Store* op) override {
        if (shared_memory_buffers.find(op->name) == shared_memory_buffers.end()) {
            add_push_stmt(op->name, op->index);
        }
        op->value.accept(this);
    }

    void visit(const Let* op) override {
        op->value.accept(this);
    }

    void visit(const LetStmt* op) override {
        ResourceAllocationBuilder builder;
        op->body.accept(&builder);
        stmts.push_back(LetStmt::make(op->name, op->value, builder.get_stmt()));
        op->value.accept(this);
    }

    void visit(const For* op) override {
        ResourceAllocationBuilder builder;
        op->body.accept(&builder);
        stmts.push_back(For::make(op->name, op->min, op->extent, op->for_type, op->device_api, builder.get_stmt()));
    }

    void visit(const IfThenElse* op) override {
        ResourceAllocationBuilder builder;
        op->then_case.accept(&builder);
        Stmt then_case = builder.get_stmt();
        builder.stmts.clear();
        op->else_case.accept(&builder);
        Stmt else_case = builder.get_stmt();
        stmts.push_back(IfThenElse::make(op->condition, then_case, else_case));
    }

    Stmt get_stmt() {
        return Block::make(stmts);
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
                thread_loop = op->body;

                ResourceAllocationBuilder builder;
                op->body.accept(&builder);
                Stmt letStmt = LetStmt::make("bank", bank, builder.get_stmt());
                // return For::make(op->name, op->min, op->extent, ForType::Serial, DeviceAPI::None, letStmt);
            } else if (ends_with(op->name, ".__bank_id_x")) {
                num_banks[0] = op->extent;
                name_banks[0] = op->name;
                bank = op->name;
            } else if (ends_with(op->name, ".__bank_id_y")) {
                num_banks[1] = op->extent;
                name_banks[1] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), op->name);
            } else if (ends_with(op->name, ".__bank_id_z")) {
                num_banks[2] = op->extent;
                name_banks[2] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), op->name);
            } else if (ends_with(op->name, ".__bank_id_w")) {
                num_banks[3] = op->extent;
                name_banks[3] = op->name;
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
        auto allocation_script = LetStmt::make("dpu_idx", inspect.bank, builder.get_stmt());

        Stmt for_copy_allocation = For::make(loop->name, loop->min, loop->extent, loop->for_type, DeviceAPI::None, loop->body);
        ThreadLoopMutator mutator(allocation_script);
        Stmt stmt_copy_to = mutator.mutate(for_copy_allocation);

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