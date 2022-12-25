#include <assert.h>
#include <stdio.h>

#include <cuda_runtime_api.h>

#include "axpy.hpp"

__global__ void sp_kernel(int n, float a, float* x, int incx, float* y, int incy)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (((idx * incx) < n) && ((idx * incy) < n))
    {
        y[idx * incy] = y[idx * incy] + a * x[idx * incx];;\
    }
}

__global__ void dp_kernel(int n, double a, double* x, int incx, double* y, int incy)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (((idx * incx) < n) && ((idx * incy) < n))
    {
        y[idx * incy] = y[idx * incy] + a * x[idx * incx];
    }
}

void saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy)
{
	float* x_gpu = nullptr, *y_gpu = nullptr;
    (cudaMalloc((void**)&x_gpu, n * sizeof(float)) == cudaSuccess);
	(cudaMalloc((void**)&y_gpu, n * sizeof(float)) == cudaSuccess);
	(cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    (cudaMemcpy(y_gpu, y, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
	int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	sp_kernel <<<num_blocks, BLOCK_SIZE>>>(n, a, x_gpu, incx, y_gpu, incy);
    (cudaDeviceSynchronize() == cudaSuccess);
	(cudaMemcpy(y, y_gpu, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    (cudaFree(x_gpu) == cudaSuccess);
	(cudaFree(y_gpu) == cudaSuccess);
}

void  daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy)
{
	double* x_gpu = nullptr, *y_gpu = nullptr;
	(cudaMalloc((void**)&x_gpu, n * sizeof(double)) == cudaSuccess);
    (cudaMalloc((void**)&y_gpu, n * sizeof(double)) == cudaSuccess);
	(cudaMemcpy(x_gpu, x, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
	(cudaMemcpy(y_gpu, y, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
	int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dp_kernel <<<num_blocks, BLOCK_SIZE>>>(n, a, x_gpu, incx, y_gpu, incy);
    (cudaDeviceSynchronize() == cudaSuccess);
    (cudaMemcpy(y, y_gpu, n * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
    (cudaFree(x_gpu) == cudaSuccess);
    (cudaFree(y_gpu) == cudaSuccess);
}
