#ifndef HALIDE_HALIDERUNTIMEUPMEM_H
#define HALIDE_HALIDERUNTIMEUPMEM_H

#ifndef HALIDE_HALIDERUNTIME_H
#include "HalideRuntime.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern const struct halide_device_interface_t *halide_upmem_device_interface();

// DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
// DPU_ASSERT(dpu_load(dpu_set, BINARY, NULL));
// DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));

// DPU_ASSERT(dpu_free(dpu_set));

extern int halide_upmem_run(void* user_context, const char* kernel_name, uint32_t nr_tasklets);
extern int halide_upmem_dpu_copy_to(void *user_context, int32_t dpu_idx, struct halide_buffer_t *buf, uint32_t offset, uint32_t size);
extern int halide_upmem_dpu_copy_from(void *user_context, int32_t dpu_idx, struct halide_buffer_t *buf, uint32_t offset, uint32_t size);
extern int halide_upmem_dpu_xfer_to(void *user_context, int32_t dpu_idx, struct halide_buffer_t *buf, uint32_t host_offset[], uint32_t dpu_offset, uint32_t size);
extern int halide_upmem_dpu_xfer_from(void *user_context, int32_t dpu_idx, struct halide_buffer_t *buf, uint32_t host_offset[], uint32_t dpu_offset, uint32_t size);
extern int halide_upmem_alloc_load(void *user_context, int32_t nr_dpus, const char* kernel_name);
extern int halide_upmem_free(void *user_context, const char* kernel_name);
extern halide_buffer_info_t * halide_upmem_info_args(void* args, uint32_t i);


#ifdef __cplusplus
}
#endif

#endif