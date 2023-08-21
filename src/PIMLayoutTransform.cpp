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
#include "Bounds.h"
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
                return stmt;
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

        set<string> shared_memory_buffer;

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
                bank = Variable::make(Int(32), op->name);
            } else if (ends_with(op->name, ".__bank_id_y")) {
                num_banks[1] = op->extent;
                name_banks[1] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), Variable::make(Int(32), op->name));
            } else if (ends_with(op->name, ".__bank_id_z")) {
                num_banks[2] = op->extent;
                name_banks[2] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), Variable::make(Int(32), op->name));
            } else if (ends_with(op->name, ".__bank_id_w")) {
                num_banks[3] = op->extent;
                name_banks[3] = op->name;
                bank = Add::make(Mul::make(bank, Expr(op->extent)), Variable::make(Int(32), op->name));
            }
            op->body.accept(this);
        }

        void visit(const ProducerConsumer* op) override {
            shared_memory_buffer.insert(op->name);
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

        vector<Stmt> stmt_copies;
        auto bounds = boxes_touched(inspect.thread_loop);
        for (auto it = bounds.begin(); it != bounds.end(); it++) {
            auto box = it->second;
            vector<Expr> offset, sizes;
            for (Interval b: box.bounds) {
                offset.push_back(b.min);
                sizes.push_back(b.max - b.min + 1);
            }
            vector<Expr> args = { Variable::make(Int(32), "dpu_idx"), Variable::make(type_of<struct halide_buffer_t *>(), it->first + ".buffer") };
            args.insert(args.end(), offset.begin(), offset.end());
            args.insert(args.end(), sizes.begin(), sizes.end());
            if (inspect.shared_memory_buffer.find(it->first) == inspect.shared_memory_buffer.end())
                stmt_copies.push_back(Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_to", args, Call::Extern)));
        }
        ThreadLoopMutator mutator(LetStmt::make("dpu_idx", inspect.bank, Block::make(stmt_copies)));
        Stmt stmt_copy_to = mutator.mutate(loop);

        Stmt stmt_free = call_extern_and_assert("halide_upmem_free", { kernel_name });

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