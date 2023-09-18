#include "AOT_result/gemv.h"
#include "HalideBuffer.h"

// g++ lesson_10*run.cpp lesson_10_halide.a -std=c++17 -I <path/to/Halide.h> -lpthread -ldl -o lesson_10_run

using namespace Halide::Runtime;

int main() {
    const int M = 1024;
    const int N = 2048;

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

    gemv_execute(infos);
    gemv_copy_from(output, infos);

    haldie_arg_infos_free(infos);


    for (int i = 0; i < M; i++) {
        printf("%d, ", output(i));
    }
}

/*
g++ source_aot_realize.cpp AOT_result/gemv_generate_host.c -g -std=c++17 -I $HALIDE_DIR/include -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o AOT_result/gemv_generate_run
./AOT_result/gemv_generate_run
*/