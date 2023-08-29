#include "gemv_test.h"

#include "HalideBuffer.h"

using namespace Halide;

int main() {
    const int M = 8192;
    const int N = 4096;

    Buffer<int> A(M, N, "WEIGHT");
    Buffer<int> x(M, "VECTOR");
    Buffer<int> output(N, "OUTPUT");

    Buffer<int> A_transformed = gemv_transform(A);
    Buffer<int> x_transformed = gemv_transform(x);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            A(m, n) = rand() % 100;
        }
    }
    for (int n = 0; n < N; n++) {
        x(n) = rand() % 100;
    }

    gemv_init(A, x, output);
    // or gemv_init(A, x, output);
    gemv_transfer_weight(A);
    gemv_transfer_vector(x);
    gemv();
    gemv_transfer_output(output);

    Buffer<int> output_transformed = gemv_transform(output);
    for (int i = 0; i < M; i++) {
        printf("%d, ", output(i));
    }
}