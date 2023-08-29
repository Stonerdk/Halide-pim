#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>

uint64_t* W;
uint64_t* W_transformed;

int main() {
    int m = 8192;
    int n = 4096;
    int T = 4;
    int mt = m / T;
    W = (uint64_t*)malloc(sizeof(uint64_t) * m * n);
    for (int i = 0; i < m * n; i++) {
        W[i] = i;
    }
    struct dpu_set_t dpu_set, dpu;
    DPU_ASSERT(dpu_alloc(4, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, "./bin/bin", NULL));

    uint32_t i = 0;

    clock_t startTime = clock();
    uint64_t offset = 0;
    for (int j = 0; j < n; j++) {
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, W + offset + i * mt));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, offset, sizeof(uint64_t) * mt, DPU_XFER_DEFAULT));
        offset += j * n;
    }
    clock_t endTime = clock();
    double duration = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("basic duration: %f seconds\n", duration);

    // transform
    startTime = clock();
    W_transformed = (uint64_t*)malloc(sizeof(uint64_t) * m * n);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < n; j++) {
            memcpy(W_transformed + (i * n + j) * mt, W + i * mt + j * m, sizeof(uint64_t) * mt); 
        }
    }
    endTime = clock();
    duration = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("transform duration - transform: %f seconds\n", duration);

    startTime = clock();
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, W + i * mt * n));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, sizeof(uint64_t) * mt * n, DPU_XFER_DEFAULT));
    endTime = clock();
    duration = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("transform duration - xfer: %f seconds\n", duration);

    DPU_ASSERT(dpu_free(dpu_set));
    free(W);
    free(W_transformed);
    return 0;
}

// transform: 0.16sec