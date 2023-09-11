#include "AOT_result/gemv_generate_host.c"
#include "HalideBuffer.h"

using namespace Halide::Runtime;

int main() {
    const int M = 128;
    const int N = 512;

    Buffer<int> A(M, N);
    Buffer<int> x(M);
    Buffer<int> output(N);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            A(m, n) = rand() % 100;
        }
    }
    for (int n = 0; n < N; n++) {
        x(n) = rand() % 100;
    }

    auto infos = halide_arg_infos({ output, A, x });

    gemv_copy_to_0(A, infos);
    gemv_copy_to_1(x, infos);

    gemv_execute();
    gemv_copy_from(output, infos);

    haldie_arg_infos_free(infos);

    // Buffer<int> output_transformed = gemv_transform(output);
    for (int i = 0; i < M; i++) {
        printf("%d, ", output(i));
    }
}

/*
g++ source_aot_realize.cpp AOT_result/gemv_generate_host.c -g -std=c++17 -I $HALIDE_DIR/include -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o AOT_result/gemv_generate_run
./AOT_result/gemv_generate_run
*/