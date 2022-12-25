
#include "cufft_task.hpp"

__global__ void convolution(const complex* a, const complex* b, complex *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx].x = a[idx].x * b[idx].x / n - a[idx].y * b[idx].y / n;
        c[idx].y = a[idx].x * b[idx].y / n + a[idx].y * b[idx].x / n;
    }
}

void fft_gpu(const complex* a, const complex* b, complex* c, int n)
{
    cufftHandle plan;
    cufftComplex* a_gpu = nullptr, *b_gpu = nullptr, *c_gpu = nullptr;
    cudaMalloc((void**)&a_gpu, sizeof(cufftComplex) * n);
    cudaMalloc((void**)&b_gpu, sizeof(cufftComplex) * n);
    cudaMalloc((void**)&c_gpu, sizeof(cufftComplex) * n);
    cudaMemcpy(a_gpu, a, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu, c, sizeof(cufftComplex) * n, cudaMemcpyHostToDevice);
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
    cufftExecC2C(plan, a_gpu, a_gpu, CUFFT_FORWARD);
    cufftExecC2C(plan, b_gpu, b_gpu, CUFFT_FORWARD);
    unsigned num_blocks = (n + FFT_BLOCK_SIZE - 1) / FFT_BLOCK_SIZE;
    convolution <<<num_blocks, FFT_BLOCK_SIZE>>> (a_gpu, b_gpu, c_gpu, n);
    cudaDeviceSynchronize();
    cufftExecC2C(plan, c_gpu, c_gpu, CUFFT_INVERSE);
    cudaMemcpy(c, c_gpu, sizeof(cufftComplex) * n, cudaMemcpyDeviceToHost);
    cufftDestroy(plan);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
