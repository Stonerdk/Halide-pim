cd /root/dev/halide-pim
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=/root/dev/llvm-install/lib/cmake/llvm -DClang_DIR=/root/dev/llvm-install/lib/clang/llvm -S . -B build \
&& cmake --build build -j 8 --parallel \
&& cmake --install ./build --prefix /root/dev/halide-pim-install \
&& cd ~/dev/halide-pim/custom_test/gemv/ \
&& g++ source.cpp -g -std=c++17 -I $HALIDE_DIR/include -I $HALIDE_DIR/tools \
-L $HALIDE_DIR/lib  -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o gemv \
&& LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./gemv 2> gemv_result.txt \
&& LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 gdb ./gemv
# && LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./gemv > gemv_result.txt