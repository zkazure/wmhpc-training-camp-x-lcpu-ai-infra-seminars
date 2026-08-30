#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                cudaGetErrorName(err), __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
        } \
        } while(0)


#define CUDA_CHECK_KERNEL() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
        CUDA_CHECK(cudaDeviceSynchronize());   \
    } while (0)

__global__ void
saxpy
(
 const float *x,
 float *y,
 int n
 )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = 2.0 * x[idx] + y[idx];
    }
}


int main(int argc, char **argv) {
    const int n = atoi(argv[1]);
    if (n == 0) {
        printf("SUM=%.0f\n", 0.0);
        return 0;
    }
    size_t bytes = (size_t) n * sizeof(float);

    float *h_y = (float *) malloc(bytes);
    float *h_x = (float *) malloc(bytes);
    for (int i = 0; i < n; ++i) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }

    float *d_y, *d_x;
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(d_x, d_y, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += h_y[i];
    }
    printf("SUM=%.0f\n", s);

    return 0;
}
