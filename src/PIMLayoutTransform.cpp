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

using namespace std; 

Expr make_shape_var(string name, const string &field, size_t dim,
                        const Buffer<> &buf, const Parameter &param) {
    ReductionDomain rdom;
    name = name + "." + field + "." + to_string(dim);
    return Variable::make(Int(32), name, buf, param, rdom);
}

class PIMLayoutTransform : public IRMutator {
public:

    PIMLayoutTransform(map<string, Stmt>& splitted_stmts): splitted_stmts(splitted_stmts) {}

    PIMLayoutTransform() {}
    
    optional<reference_wrapper<map<string, Stmt>>> splitted_stmts;

    set<string> zero_realized;

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
                bank = bank * op->extent + Variable::make(Int(32), op->name);
            } else if (ends_with(op->name, ".__bank_id_z")) {
                num_banks[2] = op->extent;
                name_banks[2] = op->name;
                bank = bank * op->extent + Variable::make(Int(32), op->name);
            } else if (ends_with(op->name, ".__bank_id_w")) {
                num_banks[3] = op->extent;
                name_banks[3] = op->name;
                bank = bank * op->extent + Variable::make(Int(32), op->name);
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
    using IRMutator::visit;
    size_t kernel_idx = 0;
    Scope<> result_scope;
    Scope<Halide::Region> intm_scope; // intm_scope ⊂ result_scope
    Scope<string> zero_realize; // intm which is defined as zer

    Scope<map<string, Box>> bounds_scope;
    Scope<Expr> let_values;
    

    set<string> shared_memory_buffer;
    bool inside_pim_loop;

    Stmt visit(const ProducerConsumer* op) override {
        result_scope.push(op->name);
        Stmt stmt = IRMutator::visit(op);
        result_scope.pop(op->name);
        return stmt;
    }

    string get_kernel_name() {
        return "kernel_" + to_string(kernel_idx);
    }

    bool is_split() {
        return splitted_stmts.has_value();
    }

    void split_merge(string name, Stmt stmt, bool tail = true) {
        if (!is_split()) return;
        auto& ref = splitted_stmts.value().get();
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

    Stmt visit(const Realize *op) override {
        zero_realize.push(op->name, op->name);
        intm_scope.push(op->name, op->bounds);
        zero_realized.insert(op->name);
        Stmt stmt = IRMutator::visit(op);
        intm_scope.pop(op->name);
        zero_realize.pop(op->name);
        return stmt;
    }

    Stmt visit(const Provide *op) override {
        if (!inside_pim_loop && zero_realize.contains(op->name)) {
            for (const auto& value: op->values) {
                if (!is_const_zero(value)) {
                    if (zero_realize.contains(op->name))
                        zero_realize.pop(op->name);
                    break;
                }
            }
        }
        return IRMutator::visit(op);
    }

    Expr visit(const Load *op) override {
        if (inside_pim_loop && zero_realize.contains(op->name)) {
            if (zero_realize.contains(op->name))
                zero_realize.pop(op->name);
        }
        return IRMutator::visit(op);
    }

    Expr var_get_extent(string vname, size_t dim) {
        return Variable::make(Int(32), vname + ".extent." + to_string(dim) + ".proposed");
    }

    pair<Stmt, Stmt> inject_copy(const For* loop, Expr bank, Stmt init_stmt) {
        string kernel_name = get_kernel_name();
        map<string, Box> bounds = bounds_scope.get(kernel_name);

        vector<Stmt> stmts_copy_to, stmts_copy_from;
        set<string> copy_to_names;

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

            vector<Expr> vec_min, vec_extent, vec_stride;

            for (size_t i = 0; i < box.size(); i++) {
                Expr extent, stride, _min;
                if (intm_scope.contains(bound_it->first)) {
                    auto region = intm_scope.get(bound_it->first);
                    extent = region[i].extent;
                    _min = region[i].min;
                    if (i == 0) stride = 1;
                    else stride = region[i - 1].extent - region[i - 1].min;
                } else {
                    string vname_extent, vname_min, vname_stride;
                    vname_extent = bound_it->first + ".extent." + to_string(i) + ".proposed";
                    vname_stride = bound_it->first + ".stride." + to_string(i) + ".proposed";
                    vname_min = bound_it->first + ".min." + to_string(i) + ".proposed";
                    if (let_values.contains(vname_extent)) 
                        extent = let_values.get(vname_extent);
                    else extent = Variable::make(Int(32), vname_extent);
                    if (let_values.contains(vname_stride))
                        stride = let_values.get(vname_stride);
                    else stride = Variable::make(Int(32), vname_stride);
                    if (let_values.contains(vname_min)) 
                        _min = let_values.get(vname_min);
                    else _min = Variable::make(Int(32), vname_min);
                }
                vec_min.push_back(_min);
                vec_extent.push_back (extent);
                vec_stride.push_back(stride);
            }
            
            size_t merge_idx = 1;
            for (size_t i = 1; i < box.size(); i++) {
                if (graph_equal(size, vec_stride[i])) {
                    size *= box_intervals[i];
                    merge_idx++;
                } else {
                    break;
                }
            }
            for (size_t i = 1; i < box.size(); i++) {
                if (i < merge_idx)
                    offset += (box[i].min - vec_min[i]) * vec_stride[i];
                else
                    offset += (Variable::make(Int(32), "ii" + to_string(i)) + box[i].min - vec_min[i]) * vec_stride[i];
            }
            vector<Expr>args = { Variable::make(Int(32), "dpu_idx"), Variable::make(type_of<struct halide_buffer_t*>(), bound_it->first + ".buffer"), offset, size };
 
            if (result_scope.contains(bound_it->first)) { 
                Stmt stmt_copy_from = Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_from", args, Call::Extern));
                stmt_copy_from = ThreadLoopMutator(LetStmt::make("dpu_idx", bank, stmt_copy_from)).mutate(loop);
                for (size_t i = merge_idx; i < box.size(); i++) {
                    stmt_copy_from = For::make("ii" + to_string(i), 0, box_intervals[i], ForType::Serial, DeviceAPI::None, stmt_copy_from);
                }
                stmts_copy_from.push_back(stmt_copy_from);
            } // realize: copy_from is invoked after loop

            if (!zero_realize.contains(bound_it->first)) {
                Stmt stmt_copy_to = Evaluate::make(Call::make(Int(32), "halide_upmem_dpu_copy_to", args, Call::Extern));
                stmt_copy_to = ThreadLoopMutator(LetStmt::make("dpu_idx", bank, stmt_copy_to)).mutate(loop);
                for (size_t i = merge_idx; i < box.size(); i++) {
                    stmt_copy_to = For::make("ii" + to_string(i), 0, box_intervals[i], ForType::Serial, DeviceAPI::None, stmt_copy_to);
                }
                if (!is_split() || result_scope.contains(bound_it->first)) {
                    stmts_copy_to.push_back(stmt_copy_to);
                } else {
                    split_merge(bound_it->first, stmt_copy_to, false);
                    copy_to_names.emplace(bound_it->first);
                }
            }
        }

        for (string name: copy_to_names) {
            split_merge(name, init_stmt, false);
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
                        lets.push({ it->first + ".min.pim." + to_string(i), Variable::make(Int(32), it->first + ".min." + to_string(i)) });
                        lets.push({ it->first + ".extent.pim." + to_string(i), Variable::make(Int(32), it->first + ".extent." + to_string(i)) });
                        lets.push({ it->first + ".stride.pim." + to_string(i), Variable::make(Int(32), it->first + ".stride." + to_string(i)) });
                    } else {
                        lets.push({it->first + ".min.pim." + to_string(i), it->second[i].min});
                        Expr extent = it->second[i].max - it->second[i].min + 1;
                        lets.push({ it->first + ".extent.pim." + to_string(i), extent });
                        lets.push({ it->first + ".stride.pim." + to_string(i), stride });
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
        shared_memory_buffer = move(inspect.shared_memory_buffer);

        Stmt stmt_alloc_load = call_extern_and_assert("halide_upmem_alloc_load", { inspect.get_all_banks(), kernel_name });
        Stmt stmt_free = call_extern_and_assert("halide_upmem_free", { kernel_name });

        auto bounds = boxes_touched(inspect.thread_loop);
        bounds_scope.push(kernel_name, bounds);

        auto [ stmt_copy_to, stmt_copy_from ] = inject_copy(loop, inspect.get_bank(), stmt_alloc_load);

        inside_pim_loop = true;
        auto result = IRMutator::visit(loop);
        inside_pim_loop = false;
        bounds_scope.pop(kernel_name);

        kernel_idx++;

        if (is_split()) {
            return Block::make({ stmt_copy_to, result, stmt_copy_from, stmt_free });
            // why not split "gemv" now? -> Inside PIM computation needs lowering pass
        } else {
            return Block::make({ stmt_alloc_load, stmt_copy_to, result, stmt_copy_from, stmt_free });
        }
    }
};

class ZeroRealizeMover: public IRMutator {
public:
    using IRMutator::visit;

    set<string>& zero_realized;
    bool after_pim_loop = false;

    ZeroRealizeMover(set<string>& zero_realized): zero_realized(zero_realized) {}

    Stmt visit(const For* op) override {
        if (ends_with(op->name, ".__bank_id_x")) {
            after_pim_loop = true;
        } 
        return IRMutator::visit(op);
    }

    Stmt visit(const Provide* op) override {
        if (zero_realized.count(op->name) > 0 && !after_pim_loop) {
            return Evaluate::make(0);
        }
        return IRMutator::visit(op);
    }
};


Stmt pim_layout_transform(Stmt s) {
    return PIMLayoutTransform().mutate(s);
}

Stmt pim_layout_transform_split(Stmt s, map<string, Stmt>& splitted_stmts) {
    PIMLayoutTransform lt(splitted_stmts);
    Stmt new_s = lt.mutate(s);
    return ZeroRealizeMover(lt.zero_realized).mutate(new_s);
}
}  // namespace Internal
}  // namespace Halide