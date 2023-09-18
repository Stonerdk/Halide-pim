cd /root/dev/halide-pim
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=/root/dev/llvm-install/lib/cmake/llvm -DClang_DIR=/root/dev/llvm-install/lib/clang/llvm -S . -B build \
&& HALIDE_WITH_EXCEPTIONS=OFF cmake --build build -j 8 --parallel \
&& cmake --install ./build --prefix /root/dev/halide-pim-install \

cd ~/dev/halide-pim/custom_test/gemv/  

g++ gemv_gen.cpp -g -std=c++17 -I $HALIDE_DIR/include -I $HALIDE_DIR/tools -L $HALIDE_DIR/lib  -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -lcurses -ldl -lrt -lz -lm -o ./AOT_result/gemv_gen
LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 ./AOT_result/gemv_gen 2> ./AOT_result/gemv_gen_log.txt
# cd ~/dev/halide-pim/custom_test/gemv/; LD_LIBRARY_PATH=$HALIDE_DIR/lib HL_DEBUG_CODEGEN=2 gdb ./AOT_result/gemv_gen

cd ~/dev/halide-pim/custom_test/gemv/  
g++ gemv_run.cpp ./AOT_result/gemv.a -g -std=c++17 -I $HALIDE_DIR/include -lpthread -ldl -lz -lm -o  AOT_result/gemv_run
./AOT_result/gemv_run > ./AOT_result/gemv_run_log.txt;
cd ~/dev/halide-pim/custom_test/gemv/ 