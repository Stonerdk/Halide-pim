#ifndef HALIDE_HALIDERUNTIMEUPMEM_H
#define HALIDE_HALIDERUNTIMEUPMEM_H

#ifndef HALIDE_HALIDERUNTIME_H
#include "HalideRuntime.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern const struct halide_device_interface_t *halide_upmem_device_interface();

extern int halide_upmem_initialize_kernels(void *user_context, void **state_ptr, const char *src, int size);
// DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
// DPU_ASSERT(dpu_load(dpu_set, BINARY, NULL));
// DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));

extern int halide_upmem_run(void *user_context,
                                   void *state_ptr,
                                   const char *entry_name,
                                   int blocksX, int blocksY, int blocksZ,
                                   int threadsX, int threadsY, int threadsZ,
                                   int shared_mem_bytes,
                                   struct halide_type_t arg_types[], 
                                   void *args[], 
                                   int8_t arg_is_buffer[]);

extern int halide_upmem_finalize_kernels(void *user_context, void *state_ptr);
// DPU_ASSERT(dpu_free(dpu_set));

extern int halide_upmem_dpu_copy_to(void *user_context, size_t dpu_idx, struct halide_buffer_t *buf, uint64_t offset, uint64_t size, void *args[]);

extern int halide_upmem_dpu_copy_from(void *user_context, size_t dpu_idx, struct halide_buffer_t *buf, uint64_t offset, uint64_t size, void *args[]);

extern int halide_upmem_dpu_xfer_to(void *user_context, size_t dpu_idx, struct halide_buffer_t *buf, uint64_t host_offset[], uint64_t dpu_offset, uint64_t size, void *args[]);

extern int halide_upmem_dpu_xfer_from(void *user_context, size_t dpu_idx, struct halide_buffer_t *buf, uint64_t host_offset[], uint64_t dpu_offset, uint64_t size, void *args[]);

extern int halide_upmem_alloc_load(void *user_context, size_t nr_dpus, const char* kernel_name);

extern int halide_upmem_free(void *user_context, const char* kernel_name);

#ifdef __cplusplus
}
#endif

#endif