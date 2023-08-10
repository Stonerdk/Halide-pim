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
        for (int i = 0; i < 4; i++) {
            num_threads[i] = num_banks[i] = 1;
        }
    }

private:
    bool found_shared = false;

    using IRVisitor::visit;

    void visit(const For *op) override {
        if (CodeGen_PIM_Dev::is_pim_var(op->name)) {
            internal_assert(is_const_zero(op->min));
        }

        if (ends_with(op->name, ".__tasklet_id_x")) {
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
        if (!CodeGen_PIM_Dev::is_pim_var(loop->name)) {
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
        string kernel_name = c_print_name(unique_name("kernel_" + loop->name));

        CodeGen_PIM_Dev *pim_codegen = cgdev[loop->device_api].get();
        user_assert(pim_codegen != nullptr)
            << "Loop is scheduled on device " << loop->device_api
            << " which does not appear in target " << target.to_string() << "\n";
        pim_codegen->add_kernel(loop, kernel_name, closure_args);

        bool runtime_run_takes_types = pim_codegen->kernel_run_takes_types();
        Type target_size_t_type = target.bits == 32 ? Int(32) : Int(64);

        vector<Expr> args, arg_types_or_sizes, arg_is_buffer;
        for (const DeviceArgument &i : closure_args) {
            Expr val;
            if (i.is_buffer) {
                val = Variable::make(Handle(), i.name + ".buffer");
            } else {
                val = Variable::make(i.type, i.name);
                val = Call::make(type_of<void *>(), Call::make_struct, {val}, Call::Intrinsic);
            }
            args.emplace_back(val);

            if (runtime_run_takes_types) {
                arg_types_or_sizes.emplace_back(((halide_type_t)i.type).as_u32());
            } else {
                arg_types_or_sizes.emplace_back(cast(target_size_t_type, i.is_buffer ? 8 : i.type.bytes()));
            }

            arg_is_buffer.emplace_back(cast<uint8_t>(i.is_buffer));
        }

        // nullptr-terminate the lists
        args.emplace_back(reinterpret(Handle(), cast<uint64_t>(0)));
        if (runtime_run_takes_types) {
            internal_assert(sizeof(halide_type_t) == sizeof(uint32_t));
            arg_types_or_sizes.emplace_back(cast<uint32_t>(0));
        } else {
            arg_types_or_sizes.emplace_back(cast(target_size_t_type, 0));
        }
        arg_is_buffer.emplace_back(cast<uint8_t>(0));

        // TODO: only three dimensions can be passed to
        // cuLaunchKernel. How should we handle blkid[3]?
        internal_assert(is_const_one(bounds.num_banks[3])) << bounds.num_banks[3] << "\n";
        debug(3) << "bounds.num_blocks[0] = " << bounds.num_banks[0] << "\n";
        debug(3) << "bounds.num_blocks[1] = " << bounds.num_banks[1] << "\n";
        debug(3) << "bounds.num_blocks[2] = " << bounds.num_banks[2] << "\n";
        debug(3) << "bounds.num_threads[0] = " << bounds.num_threads[0] << "\n";

        string api_unique_name = pim_codegen->api_unique_name();
        vector<Expr> run_args = {
            get_state_var(api_unique_name),
            kernel_name,
            Expr(bounds.num_banks[0]),
            Expr(bounds.num_banks[1]),
            Expr(bounds.num_banks[2]),
            Expr(bounds.num_threads[0]),
            Expr(0),
            Call::make(Handle(), Call::make_struct, arg_types_or_sizes, Call::Intrinsic),
            Call::make(Handle(), Call::make_struct, args, Call::Intrinsic),
            Call::make(Handle(), Call::make_struct, arg_is_buffer, Call::Intrinsic),
        };
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
            cgdev[DeviceAPI::UPMEM].get()->compile_to_src();
            // no need initialize_kernel & destructor. do this behavior right before/after loops
        }

        return result;
    }
};

}  // namespace

Stmt inject_pim_offload(const Stmt &s, const Target &host_target) {
    return InjectGpuOffload(host_target).inject(s);
}

}  // namespace Internal
}  // namespace Halide
