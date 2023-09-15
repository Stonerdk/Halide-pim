#include "Halide.h"

using namespace Halide;

int main() {
    Var x("x"), y("y"), yb("yb"), xc("xc"), yc("yc"), yi("yi"), xi("xi");
    Var yb_("yb_"), xcyc("xcyc_");
    RVar xc_("xc_"), xi_("xi_");

    Func R("R"), gemv("gemv");

    const int M = 512;
    const int N = 1024;

    // Buffer<int> weight(M, N, "weight");
    // Buffer<int> vector(M, "vector");
    // Buffer<int> output(N, "output");

    ImageParam weight(type_of<int>(), 2, "weight");
    ImageParam vector(type_of<int>(), 1, "vector");
    ImageParam output(type_of<int>(), 1, "output");

    // phase 1. split xcyc as bank, reduce finally
    // R(y) = 0;

    int32_t T = 16;
    RDom r(0, weight.dim(0).extent());

    R(y) += weight(r, y) * vector(r);
    
    R.compute_root().update().split(r, xc_, xi_, T);
    gemv = R.update().rfactor({{xc_, xc}});
    gemv.compute_root().update().split(y, yc, yb_, T * T).split(yb_, yb, yi, T);
    gemv.update().reorder(yi, yb, xc, yc).pim_thread(yb).fuse(yc, xc, xcyc).pim_bank(xcyc);

    // gemv.print_loop_nest();

    //Target target = get_host_target();
    Target target = get_host_target().with_feature(Target::UPMEM).with_feature(Target::NoAsserts);
    // R.compile_to_c("AOT_result/gemv_generate", {weight, vector}, "gemv", target);
    R.compile_to_upmem_libraries("AOT_result/gemv_generate_hard", {weight, vector}, "gemv", target);

    return 0;
}

/* 
g++ source_aot.cpp -g -std=c++17 -I $HALIDE_DIR/include -I $HALIDE_DIR/tools \
-L $HALIDE_DIR/lib  -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o gemv_generate
LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./gemv_generate 2> ./AOT_result/gemv_generate_result.txt
*/
