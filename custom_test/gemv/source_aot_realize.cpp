#include "HalideBuffer.h"

using namespace Halide;

int main() {
    const int M = 8192;
    const int N = 4096;

    Halide::Runtime::Buffer<int> A(M, N);
    Halide::Runtime::Buffer<int> x(M);
    Halide::Runtime::Buffer<int> output(N);

    const halide_dimension_t *dims[] = { A.raw_buffer()->dim, x.raw_buffer()->dim, output.raw_buffer()->dim };

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            A(m, n) = rand() % 100;
        }
    }
    for (int n = 0; n < N; n++) {
        x(n) = rand() % 100;
    }

    // gemv_init(A, x, output);
    // gemv_init(output);

    halide_buffer_info_t * infos = halide_buffer_get_info({ A, x, output });


    gemv(A, x, output);

    // Buffer<int> output_transformed = gemv_transform(output);
    for (int i = 0; i < M; i++) {
        printf("%d, ", output(i));
    }
}