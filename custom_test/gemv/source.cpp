#include "Halide.h"

using namespace Halide;

int main() {

    const int M = 8192;
    const int N = 4096;

    Var i("i"), j("j"), block("block"), inner_loop("inner_loop"), thread("thread");

    Buffer<int> A(N, M, "weight");
    Buffer<int> x(N, "vector");
    Buffer<int> output(2049, "output");

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            A(n, m) = rand() % 100;
        }
    }
    for (int n = 0; n < N; n++) {
        x(n) = rand() % 100;
    }

    Func gemv("gemv");
    Func intermediate("intermediate");

    RDom r(0, N);
    // intermediate(i, j) = A(i, j) * x(j);
    gemv(i) = sum(A(r, i) * x(r));

    gemv.split(i, block, thread, 2048);
    gemv.split(thread, thread, inner_loop, 128);

    gemv.pim_bank(block);
    gemv.pim_thread(thread);

    Target target = get_host_target().with_feature(Target::UPMEM);
    gemv.compile_to_c("gemv.c", {A, x}, "gemv", target);

    //gemv.realize(output, target);

    return 0;
}

/* 
g++ source.cpp -g -std=c++17 -I $HALIDE_DIR/include -I $HALIDE_DIR/tools \
-L $HALIDE_DIR/lib  -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o gemv
LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./gemv 2> gemv_result.txt
*/
