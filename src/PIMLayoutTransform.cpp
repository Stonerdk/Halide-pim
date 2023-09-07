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
#include "IREquality.h"

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
using std::stack;
using std::reverse;

Expr make_shape_var(string name, const string &field, size_t dim,
                        const Buffer<> &buf, const Parameter &param) {
    ReductionDomain rdom;
    name = name + "." + field + "." + std::to_string(dim);
    return Variable::make(Int(32), name, buf, param, rdom);
}

class PIMLayoutTransform : public IRMutator {
public:

    PIMLayoutTransform(
        std::map<std::string, Stmt>* splitted_stmts, 
        const std::vector<Argument> &args,
        const std::vector<Function> &outputs):
        splitted_stmts(splitted_stmts), args(args), outputs(outputs) {}
    
    map<string, Stmt>* splitted_stmts ;
    const std::vector<Argument> &args;
    const std::vector<Function> &outputs;
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
    // Scope<> reduction_scope;
    Scope<map<string, Box>> bounds_scope;
    Scope<Expr> let_values;
    set<string> shared_memory_buffer;

    Stmt visit(const ProducerConsumer* op) override {
        // reduction_scope.push(op->name);
        Stmt stmt = IRMutator::visit(op);
        // reduction_scope.pop(op->name);
        return stmt;
    }

    string get_kernel_name() {
        return "kernel_" + std::to_string(kernel_idx);
    }


    void split_merge(string name, Stmt stmt, bool tail = true) {
        if (splitted_stmts == nullptr) return;
        internal_assert(splitted_stmts != nullptr);
        map<string, Stmt>& ref = *splitted_stmts;

        if (ref.find(name) != ref.end()) {
            if (tail) {
                ref[name] = Block::make(ref[name], stmt);
            } else {
                ref[name] = Block::make(stmt, ref[name]);
            }
        } else {
            ref[name] = stmt;
        }

    }

    Stmt visit(const LetStmt* op) override {
        let_values.push(op->name, op->value);
        Stmt stmt = IRMutator::visit(op);
        let_values.pop(op->name);
        return stmt;
    }

    pair<Stmt, Stmt> inject_copy(const For* loop, Expr bank, Stmt init_stmt) {
        string kernel_name = get_kernel_name();
        map<string, Box> bounds = bounds_scope.get(kernel_name);

        vector<Stmt> stmts_copy_to, stmts_copy_from;
        map<string, bool> bounds_in;

        for (const auto& arg: args) {
            if (arg.is_buffer() && bounds.find(arg.name) != bounds.end()) {
                bounds_in[arg.name] = true;
            }
        }

        for (const auto& output: outputs) {
            if (bounds.find(output.name()) != bounds.end()) {
                bounds_in[output.name()] = false;
            }
        }

        for (auto bound_it = bounds.begin(); bound_it != bounds.end(); bound_it++) {
            if (shared_memory_buffer.find(bound_it->first) != shared_memory_buffer.end()) {
                continue;
            }
            vector<Expr> box_intervals;
            
            Box box = bound_it->second;
            for (size_t i = 0; i < box.size(); i++) {
                box_intervals.push_back(simplify(box[i].max - box[i].min + 1));
            }
            Expr offset = box[0].min;
            Expr size = box_intervals[0];
            
            size_t merge_idx = 1;
            for (size_t i = 1; i < box.size(); i++) {
                string vname_extent = bound_it->first + ".extent." + std::to_string(i - 1) + ".proposed";
                string vname_min = bound_it->first + ".min." + std::to_string(i - 1) + ".proposed";

                Expr extent, _min, stride;
                if (let_values.contains(vname_extent)) extent = let_values.get(vname_extent);
                else extent = Variable::make(Int(32), vname_extent);
                if (let_values.contains(vname_min)) _min = let_values.get(vname_min);
                else _min = Variable::make(Int(32), vname_min);
                stride = extent - _min; 
                
                debug(2) << stride << " VS " << size << "\n";
                if (graph_equal(size, stride)) {
                    size *= box_intervals[i];
                    merge_idx++;
                } else {
                    break;
                }
            }
            for (size_t i = 1; i < box.size(); i++) {
                Expr stride = Variable::make(Int(32), bound_it->first + ".stride." + std::to_string(i) + ".proposed");
                Expr _min = Variable::make(Int(32), bound_it->first + ".min." + std::to_string(i) + ".proposed");
                if (i < merge_idx)
                    offset += (box[i].min - _min) * stride;
                else
                    offset += (Variable::make(Int(32), "ii" + std::to_string(i)) + box[i].min - _min) * stride;
            }
            vector<Expr>args = { 
                Variable::make(Int(32), "dpu_idx"), 
                Variable::make(type_of<void *>(), bound_it->first),
                offset,
                size
            };

            internal_assert(bounds_in.find(bound_it->first) != bounds_in.end()) << "For now, input/output buffer for PIM should be specified either in args or outputs.\n";
            if (bounds_in[bound_it->first]) {
                Stmt stmt_copy_to = Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_to", args, Call::Extern));
                stmt_copy_to = ThreadLoopMutator(LetStmt::make("dpu_idx", bank, stmt_copy_to)).mutate(loop);
                for (size_t i = merge_idx; i < box.size(); i++) {
                    stmt_copy_to = For::make("ii" + std::to_string(i), 0, box_intervals[i], ForType::Serial, DeviceAPI::None, stmt_copy_to);
                }
                stmt_copy_to = Block::make(init_stmt, stmt_copy_to);
                split_merge(bound_it->first, stmt_copy_to, false);
                stmts_copy_to.push_back(stmt_copy_to);
            }
            else { // bounds_in[bound_it->first] = false;
                Stmt stmt_copy_from = Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_from", args, Call::Extern));
                stmt_copy_from = ThreadLoopMutator(LetStmt::make("dpu_idx", bank, stmt_copy_from)).mutate(loop);
                for (size_t i = merge_idx; i < box.size(); i++) {
                    stmt_copy_from = For::make("ii" + std::to_string(i), 0, box_intervals[i], ForType::Serial, DeviceAPI::None, stmt_copy_from);
                }
                stmts_copy_from.push_back(stmt_copy_from);
            }
        }
        if (stmts_copy_to.empty()) stmts_copy_to = { Evaluate::make(0) };
        if (stmts_copy_from.empty()) stmts_copy_from = { Evaluate::make(0) };
        return {
            Block::make(stmts_copy_to),
            Block::make(stmts_copy_from)
        };
    }

    Stmt visit(const For* loop) override {
        if (ends_with(loop->name, ".__thread_id_x")) {
            stack<pair<string, Expr>> lets;
            map<string, Box> bounds = bounds_scope.get(get_kernel_name());
            for (auto it = bounds.begin(); it != bounds.end(); it++) {
                Expr stride = 1;
                for (size_t i = 0; i < it->second.size(); i++) {
                    if (shared_memory_buffer.find(it->first) != shared_memory_buffer.end()) {
                        lets.push({ it->first + ".min.pim." + std::to_string(i), 
                            Variable::make(Int(32), it->first + ".min." + std::to_string(i)) });
                        lets.push({ it->first + ".extent.pim." + std::to_string(i), 
                            Variable::make(Int(32), it->first + ".extent." + std::to_string(i)) });
                        lets.push({ it->first + ".stride.pim." + std::to_string(i), 
                            Variable::make(Int(32), it->first + ".stride." + std::to_string(i)) });
                    } else {
                        lets.push({it->first + ".min.pim." + std::to_string(i), it->second[i].min});
                        Expr extent = it->second[i].max - it->second[i].min + 1;
                        lets.push({ it->first + ".extent.pim." + std::to_string(i), extent });
                        lets.push({ it->first + ".stride.pim." + std::to_string(i), stride });
                        stride *= extent;
                    }
                }
            }
            Stmt result = IRMutator::visit(loop);
            while (lets.size() > 0) {
                auto let = lets.top();
                result = LetStmt::make(let.first, let.second, result);
                lets.pop();
            }
            return result;
        }

        if (!ends_with(loop->name, ".__bank_id_x")) return IRMutator::visit(loop);

        string kernel_name = get_kernel_name();

        InspectResourceAllocation inspect;
        loop->accept(&inspect);
        shared_memory_buffer = std::move(inspect.shared_memory_buffer);

        Stmt stmt_alloc_load = call_extern_and_assert("halide_upmem_alloc_load", { inspect.get_all_banks(), kernel_name });
        Stmt stmt_free = call_extern_and_assert("halide_upmem_free", { kernel_name });

        auto bounds = boxes_touched(inspect.thread_loop);
        bounds_scope.push(kernel_name, bounds);

        auto [ stmt_copy_to, stmt_copy_from ] = inject_copy(loop, inspect.get_bank(), stmt_alloc_load);

        auto result = IRMutator::visit(loop);
        bounds_scope.pop(kernel_name);

        kernel_idx++;

        if (splitted_stmts) {
            return Block::make({ result, stmt_copy_from, stmt_free });
            // why not split "gemv" now? -> Inside PIM computation needs lowering pass
        } else {
            return Block::make({ stmt_alloc_load, stmt_copy_to, result, stmt_copy_from, stmt_free });
        }
    }
};

Stmt pim_layout_transform_split(Stmt s,
    const std::vector<Argument> &args,
    const std::vector<Function> &outputs,
    std::map<std::string, Stmt>& splitted_stmts) {
    return PIMLayoutTransform(&splitted_stmts, args, outputs).mutate(s);
}

Stmt pim_layout_transform(Stmt s, const std::vector<Argument> &args, const std::vector<Function> &outputs) {
    return PIMLayoutTransform(nullptr, args, outputs ).mutate(s);
}

}  // namespace Internal
}  // namespace Halide