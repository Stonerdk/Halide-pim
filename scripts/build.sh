cd /root/dev/halide-pim
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLLVM_DIR=/root/dev/llvm-install/lib/cmake/llvm -DClang_DIR=/root/dev/llvm-install/lib/clang/llvm -S . -B build
cmake --build build
cmake --install ./build --prefix /root/dev/halide-pim-install

cd custom_test
./custom_test.sh
