#include "Halide.h"

using namespace Halide;

int main() {
    Var x("x"), y("y"), yb("yb"), xo("xo"), yo("yo"), yi("yi"), xi("xi"), B("B");
    RVar xo_("xo_"), xi_("xi_");

    Func R("R"), gemv("gemv");

    const int M = 512;
    const int N = 1024;

    ImageParam weight(type_of<int>(), 2, "weight");
    ImageParam vector(type_of<int>(), 1, "vector");
    ImageParam output(type_of<int>(), 1, "output");

    int32_t T = 16;
    RDom r(0, weight.dim(0).extent());

    R(y) += weight(r, y) * vector(r);
    R.compute_root()
        .update()
        .split(r, xo_, xi_, T * T);
    gemv = R.update()
        .rfactor({{ xo_, xo }});
    gemv.compute_root()
        .update()
        .split(y, yo, yb, T * T)
        .split(yb, yb, yi, T);
    gemv.update()
        .reorder(yi, yb, xo, yo)
        .pim_thread(yb)
        .fuse(yo, xo, B)
        .pim_bank(B);

    Target target = get_host_target()
        .with_feature(Target::UPMEM)
        .with_feature(Target::UPMEM_lt_split)
        .with_feature(Target::NoAsserts)
        .with_feature(Target::Debug);
    R.compile_to_static_library("AOT_result/gemv", {weight, vector}, "gemv", target);

    return 0;
}

/* 
g++ source_aot.cpp -g -std=c++17 -I $HALIDE_DIR/include -I $HALIDE_DIR/tools \
-L $HALIDE_DIR/lib  -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o gemv_generate
LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./gemv_generate 2> ./AOT_result/gemv_generate_result.txt
*/
