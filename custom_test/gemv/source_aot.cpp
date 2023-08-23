#include "Halide.h"

using namespace Halide;

int main() {

    Var i("i"), j("j"), block("block"), inner_loop("inner_loop"), thread("thread");
    ImageParam A(type_of<int>(), 2, "weight");
    ImageParam x(type_of<int>(), 1, "vector");
    ImageParam output(type_of<int>(), 1, "output");

    Func gemv("gemv");
    Func intermediate("intermediate");

    RDom r(0, A.dim(1).extent());
    intermediate(i, j) = A(i, j) * x(j);
    gemv(i) = sum(intermediate(i, r));

    gemv.split(i, block, thread, 2048);
    gemv.split(thread, thread, inner_loop, 128);

    gemv.pim_bank(block);
    gemv.pim_thread(thread);

    Target target = get_host_target().with_feature(Target::UPMEM);
    // gemv.compile_to_static_library("gemv_test", {A, x, output}, "gemv", target);
    gemv.compile_to_c("gemv_test.c", {A, x, output}, "gemv", target);

    return 0;
}

/* 
g++ source_aot.cpp -g -std=c++17 -I $HALIDE_DIR/include -I $HALIDE_DIR/tools \
-L $HALIDE_DIR/lib  -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o gemv_generate
LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./gemv_generate 2> gemv_generate_result.txt
*/
