#include <memory>

#include "Closure.h"
#include "CodeGen_D3D12Compute_Dev.h"
#include "CodeGen_PIM_Dev.h"
#include "CodeGen_UPMEM_DPU.h"
#include "ExprUsesVar.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "InjectHostDevBufferCopies.h"
#include "OffloadPIMLoops.h"
#include "Util.h"

namespace Halide {
namespace Internal {

using std::map;
using std::string;
using std::unique_ptr;
using std::vector;
using std::pair;
using std::ostringstream;

namespace {

// Sniff the contents of a kernel to extracts the bounds of all the
// thread indices (so we know how many threads to launch), and the
// amount of shared memory to allocate.
class ExtractBounds : public IRVisitor {
public:
    Expr num_threads[1];
    Expr num_banks[4];

    ExtractBounds() {
        num_threads[0] = 1;
        for (int i = 0; i < 4; i++) {
            num_banks[i] = 1;
        }
    }

private:
    using IRVisitor::visit;

    void visit(const For *op) override {
        if (CodeGen_PIM_Dev::is_pim_var(op->name)) {
            internal_assert(is_const_zero(op->min));
        }

        if (ends_with(op->name, ".__thread_id_x")) {
            num_threads[0] = op->extent;
        } else if (ends_with(op->name, ".__bank_id_x")) {
            num_banks[0] = op->extent;
        } else if (ends_with(op->name, ".__bank_id_y")) {
            num_banks[1] = op->extent;
        } else if (ends_with(op->name, ".__bank_id_z")) {
            num_banks[2] = op->extent;
        } else if (ends_with(op->name, ".__bank_id_w")) {
            num_banks[3] = op->extent;
        }

        op->body.accept(this);
    }
};

class InjectGpuOffload : public IRMutator {
    /** Child code generator for device kernels. */
    map<DeviceAPI, unique_ptr<CodeGen_PIM_Dev>> cgdev;

    map<string, bool> state_needed;

    const Target &target;

    size_t loop_idx = 0;
    Expr get_state_var(const string &name) {
        // Expr v = Variable::make(type_of<void *>(), name);
        state_needed[name] = true;
        return Load::make(type_of<void *>(), name, 0,
                          Buffer<>(), Parameter(), const_true(), ModulusRemainder());
    }

    Expr make_state_var(const string &name) {
        auto storage = Buffer<void *>::make_scalar(name + "_buf");
        storage() = nullptr;
        Expr buf = Variable::make(type_of<halide_buffer_t *>(), storage.name() + ".buffer", storage);
        return Call::make(Handle(), Call::buffer_get_host, {buf}, Call::Extern);
    }

    // Create a Buffer containing the given vector, and return an
    // expression for a pointer to the first element.
    Expr make_buffer_ptr(const vector<char> &data, const string &name) {
        Buffer<uint8_t> code((int)data.size(), name);
        memcpy(code.data(), data.data(), (int)data.size());
        Expr buf = Variable::make(type_of<halide_buffer_t *>(), name + ".buffer", code);
        return Call::make(Handle(), Call::buffer_get_host, {buf}, Call::Extern);
    }

    using IRMutator::visit;

    Stmt visit(const For *loop) override {
        if (!CodeGen_PIM_Dev::is_pim_var(loop->name) || (loop->device_api != DeviceAPI::UPMEM && loop->device_api != DeviceAPI::Default_PIM)) {
            return IRMutator::visit(loop);
        }

        // We're in the loop over outermost block dimension
        debug(2) << "Kernel launch: " << loop->name << "\n";

        internal_assert(loop->device_api != DeviceAPI::Default_GPU)
            << "A concrete device API should have been selected before codegen.";

        ExtractBounds bounds;
        loop->accept(&bounds);
        debug(2) << "Kernel bounds: ("
                 << bounds.num_threads[0] << ") threads, ("
                 << bounds.num_banks[0] << ", "
                 << bounds.num_banks[1] << ", "
                 << bounds.num_banks[2] << ", "
                 << bounds.num_banks[3] << ") banks\n";

        HostClosure c;
        c.include(loop->body, loop->name);
        vector<DeviceArgument> closure_args = c.arguments();

        sort(closure_args.begin(), closure_args.end(),
             [](const DeviceArgument &a, const DeviceArgument &b) {
                 if (a.is_buffer == b.is_buffer) {
                     return a.type.bits() > b.type.bits();
                 } return a.is_buffer > b.is_buffer;
             });

        // compile the kernel
        string kernel_name = "kernel_" + std::to_string(loop_idx);
        internal_assert(loop_idx == 0);

        CodeGen_PIM_Dev *pim_codegen = cgdev[loop->device_api].get();
        user_assert(pim_codegen != nullptr)
            << "Loop is scheduled on device " << loop->device_api
            << " which does not appear in target " << target.to_string() << "\n";

        Stmt body = loop->body;
        while (body.as<For>()) {
            const For* body_for = body.as<For>();
            if (CodeGen_PIM_Dev::is_pim_var(body_for->name)) {
                body = body_for->body;
            }
        }

        // pim_codegen->add_kernel(body, kernel_name, closure_args);

        // TODO: only three dimensions can be passed to
        // cuLaunchKernel. How should we handle blkid[3]?
        internal_assert(is_const_one(bounds.num_banks[3])) << bounds.num_banks[3] << "\n";
        debug(3) << "bounds.num_blocks[0] = " << bounds.num_banks[0] << "\n";
        debug(3) << "bounds.num_blocks[1] = " << bounds.num_banks[1] << "\n";
        debug(3) << "bounds.num_blocks[2] = " << bounds.num_banks[2] << "\n";
        debug(3) << "bounds.num_threads[0] = " << bounds.num_threads[0] << "\n";

        string api_unique_name = pim_codegen->api_unique_name();
        vector<Expr> run_args = {
            kernel_name,
            Expr(bounds.num_threads[0]),
        };
        loop_idx ++;
        return call_extern_and_assert("halide_" + api_unique_name + "_run", run_args);
    }


public:
    InjectGpuOffload(const Target &target)
        : target(target) {
        if (target.has_feature(Target::UPMEM)) {
             cgdev[DeviceAPI::UPMEM] = std::make_unique<CodeGen_UPMEM_DPU_Dev>(target);
        }
        internal_assert(!cgdev.empty()) << "Requested unknown GPU target: " << target.to_string() << "\n";
    }

    Stmt inject(const Stmt &s) {
        for (auto &i : cgdev) {
            i.second->init_module();
        }

        Stmt result = mutate(s);

        if (target.has_feature(Target::UPMEM)) {
            debug(2) << "Mocking compiliation...\n";
            // cgdev[DeviceAPI::UPMEM].get()->compile_to_src();
            // no need initialize_kernel & destructor. do this behavior right before/after loops
        }

        return result;
    }
};

}  // namespace

Stmt inject_pim_offload(const Stmt &s, const Target &host_target) {
    return InjectGpuOffload(host_target).inject(s);
}

namespace {
class ExecuteSplitter: public IRMutator {
public:
    ExecuteSplitter(bool after): after(after) {}
    using IRMutator::visit;
    using IRMutator::mutate;

    bool found = false;
    bool started = false;
    bool after = false;

    Stmt mutate(const Stmt &stmt) override {
        if (!started) return IRMutator::mutate(stmt);
        if (found) return after ? stmt : Evaluate::make(0);
        Stmt res = IRMutator::mutate(stmt);
        if (!found) return after ? Evaluate::make(0) : stmt;
        return res;
    }

private:
    Stmt visit(const ProducerConsumer* op) override {
        started = true;
        return IRMutator::visit(op);
    }

    Stmt visit(const For* op) override {
        auto res = IRMutator::visit(op);
        if (!CodeGen_PIM_Dev::is_pim_var(op->name) || (op->device_api != DeviceAPI::UPMEM && op->device_api != DeviceAPI::Default_PIM)) {
            return res;
        }
        found = true;
        if (after) return Evaluate::make(0);
        return op;
    }

    Stmt visit(const LetStmt* op) override {
        const Call* c = op->value.as<Call>();
        if (!after && c && starts_with(c->name, "_halide_buffer_get_")) {
            return mutate(op->body);
        }
        return IRMutator::visit(op);
    }
};

}

Stmt inject_pim_offload_split(const Stmt &s, const Target &host_target, Stmt& exec_stmt) {
    exec_stmt = ExecuteSplitter(false).mutate(s);
    exec_stmt = InjectGpuOffload(host_target).inject(exec_stmt);
    debug(2) << "\nexec_stmt:\n" << exec_stmt << "\n";

    return ExecuteSplitter(true).mutate(s);
}

}  // namespace Internal
}  // namespace Halide
