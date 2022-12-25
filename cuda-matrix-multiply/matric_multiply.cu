#include <cuda_runtime_api.h>

#include "matrix_multiply.hpp"

const unsigned BLOCK_SIZE = 16;

__global__ void mmult(const float* a, const float* b, float* c, float alpha, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < n && idy < n) {
		c[idy * n + idx] = a[idy * n + idx] + alpha * b[idx * n + idy];
	}
}

void task_gpu(const float* a, const float* b, float* c, float alpha, int n) {
	float* a_gpu = nullptr, *b_gpu = nullptr, *c_gpu = nullptr;
	cudaMalloc((void**)&a_gpu, sizeof(float) * n * n);
	cudaMalloc((void**)&b_gpu, sizeof(float) * n * n);
	cudaMalloc((void**)&c_gpu, sizeof(float) * n * n);
	cudaMemcpy(a_gpu, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu, c, n * n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block_size_3d(BLOCK_SIZE, BLOCK_SIZE), num_blocks_3d;
	num_blocks_3d.x = (n * n + block_size_3d.x - 1) / block_size_3d.x;
	num_blocks_3d.y = (n * n + block_size_3d.y - 1) / block_size_3d.y;
	mmult <<<num_blocks_3d, block_size_3d>>>(a_gpu, b_gpu, c_gpu, alpha, n);
	cudaDeviceSynchronize();
	cudaMemcpy(c, c_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);
}
