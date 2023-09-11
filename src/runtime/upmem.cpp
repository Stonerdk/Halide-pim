#include "HalideRuntimeUPMEM.h"

#include "device_buffer_utils.h"
#include "device_interface.h"
#include "runtime_internal.h"
namespace Halide {
namespace Runtime {
namespace Internal {
namespace UPMEM {

}
}
}
}


using namespace Halide::Runtime::Internal::UPMEM;


extern "C" {

WEAK int halide_upmem_device_malloc(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_free(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_sync(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_release(void *user_context) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_copy_to_host(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_copy_to_device(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_and_host_malloc(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_and_host_free(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_buffer_copy(void *user_context, struct halide_buffer_t *src, const struct halide_device_interface_t *dst_device_interface, struct halide_buffer_t *dst) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_crop(void *user_context, const struct halide_buffer_t *src, struct halide_buffer_t *dst) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_slice(void *user_context, const struct halide_buffer_t *src, int slice_dim, int slice_pos, struct halide_buffer_t *dst) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_device_release_crop(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_wrap_native(void *user_context, struct halide_buffer_t *buf, uint64_t handle) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_detach_native(void *user_context, struct halide_buffer_t *buf) { return halide_error_code_unimplemented; }
WEAK int halide_upmem_compute_capability(void *user_context, int *major, int *minor) { return halide_error_code_unimplemented; }



halide_buffer_info_t * halide_upmem_info_args(void* args, uint8_t i) {
    return ((halide_buffer_info_t**)args)[i];
}

WEAK int halide_upmem_dpu_copy_to(void *user_context,
    int32_t dpu_idx, struct halide_buffer_t *buf, 
    uint64_t offset, uint64_t size) {
    debug(user_context) << "halide_upmem_dpu_copy_to " << dpu_idx << " " << buf << " " << offset << " " << size << "\n";
    return 0;
}

WEAK int halide_upmem_run(void *user_context, const char* kernel_name, uint32_t nr_tasklets) {
    debug(user_context) << "halide_upmem_run " << kernel_name << "\n";
    return 0;
}

WEAK int halide_upmem_dpu_copy_from(void *user_context,
    int32_t dpu_idx, struct halide_buffer_t *buf, 
    uint64_t offset, uint64_t size) {
    debug(user_context) << "halide_upmem_dpu_copy_from " << dpu_idx << " " << buf << " " << offset << " " << size << "\n";
    return 0;
}

WEAK int halide_upmem_dpu_xfer_to(void *user_context,
    int32_t dpu_idx, struct halide_buffer_t *buf, 
    uint64_t host_offset[], uint64_t dpu_offset, uint64_t size) {
    return halide_error_code_unimplemented;
}

WEAK int halide_upmem_dpu_xfer_from(void *user_context,
    int32_t dpu_idx, struct halide_buffer_t *buf, 
    uint64_t host_offset[], uint64_t dpu_offset, uint64_t size) {
    return halide_error_code_unimplemented;
}

WEAK int halide_upmem_alloc_load(void *user_context, int32_t nr_dpus, const char* kernel_name) {
    debug(user_context) << "halide_upmem_alloc_load " << nr_dpus << " " << kernel_name << "\n";
    return 0;
}

WEAK int halide_upmem_free(void *user_context, const char* kernel_name) {
    debug(user_context) << "halide_upmem_free " << kernel_name << "\n";
    return 0;
}


namespace {

WEAK __attribute__((constructor)) void register_upmem_allocation_pool() {

}

WEAK __attribute__((destructor)) void halide_upmem_cleanup() {
    halide_upmem_device_release(nullptr);
}

// --------------------------------------------------------------------------

}  // namespace

// --------------------------------------------------------------------------

}  // extern "C" linkage

// --------------------------------------------------------------------------

namespace Halide {
namespace Runtime {
namespace Internal {
namespace UPMEM {
// --------------------------------------------------------------------------

WEAK halide_device_interface_impl_t upmem_device_interface_impl = {
    halide_use_jit_module,
    halide_release_jit_module,
    halide_upmem_device_malloc,
    halide_upmem_device_free,
    halide_upmem_device_sync,
    halide_upmem_device_release,
    halide_upmem_copy_to_host,
    halide_upmem_copy_to_device,
    halide_upmem_device_and_host_malloc,
    halide_upmem_device_and_host_free,
    halide_upmem_buffer_copy,
    halide_upmem_device_crop,
    halide_upmem_device_slice,
    halide_upmem_device_release_crop,
    halide_upmem_wrap_native,
    halide_upmem_detach_native,
};

WEAK halide_device_interface_t upmem_device_interface = {
    halide_device_malloc,
    halide_device_free,
    halide_device_sync,
    halide_device_release,
    halide_copy_to_host,
    halide_copy_to_device,
    halide_device_and_host_malloc,
    halide_device_and_host_free,
    halide_buffer_copy,
    halide_device_crop,
    halide_device_slice,
    halide_device_release_crop,
    halide_device_wrap_native,
    halide_device_detach_native,
    halide_upmem_compute_capability,
    &upmem_device_interface_impl};

// --------------------------------------------------------------------------

}  // namespace upmem
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide
