#include <cuda_runtime_api.h>

#include "gemm.hpp"

const unsigned MATRIX_BLOCK_SIZE = 16;

__global__ void naive_multiply(float* a, float* b, float *c, int m, int p, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if ((idx < n) && (idy < m)) {
		float s = 0;
		for (int k = 0; k < p; k++) {
			s += a[idy*p + k] * b[k*n + idx];
		}
		c[idy*n + idx] = s;
	}
}

__global__ void block_multiply(float* a, float* b, float *c, int m, int p, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	const int sub_matrix_num = p / MATRIX_BLOCK_SIZE;
	__shared__ float a_block[MATRIX_BLOCK_SIZE][MATRIX_BLOCK_SIZE];
	__shared__ float b_block[MATRIX_BLOCK_SIZE][MATRIX_BLOCK_SIZE];
	if ((idx < n) && (idy < m) && (sub_matrix_num > 0)) {
		float s = 0;
		for (int i = 0; i < sub_matrix_num; i++) {
			a_block[threadIdx.x][threadIdx.y] = a[idy * p + threadIdx.x
				+ (i * MATRIX_BLOCK_SIZE)];
			b_block[threadIdx.x][threadIdx.y] = b[(threadIdx.y + i * MATRIX_BLOCK_SIZE)
				* n + idx];
			__syncthreads();
			for (int k = 0; k < MATRIX_BLOCK_SIZE; k++) {
				s += a_block[k][threadIdx.y] * b_block[threadIdx.x][k];
			}
			__syncthreads();
		}
		c[idy * n + idx] = s;
	}
}

float naive_multiply_gpu(float* a, float* b, float* c, int m, int p, int n) {
	float time = 0.f;
	float* a_gpu = nullptr, *b_gpu = nullptr, *c_gpu = nullptr;
	cudaMalloc((void**)&a_gpu, m * p * sizeof(float));
	cudaMalloc((void**)&b_gpu, p * n * sizeof(float));
	cudaMalloc((void**)&c_gpu, m * n * sizeof(float));
	cudaMemcpy(a_gpu, a, m * p * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, p * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu, c, m * n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block_size_3d(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE), num_blocks_3d;
	num_blocks_3d.x = (m * n + block_size_3d.x - 1) / block_size_3d.x;
	num_blocks_3d.y = (m * n + block_size_3d.y - 1) / block_size_3d.y;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	naive_multiply <<< num_blocks_3d, block_size_3d >>> (a_gpu, b_gpu, c_gpu, m, p, n);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaMemcpy(c, c_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);
	return time / 1000;
}

float block_multiply_gpu(float* a, float* b, float* c, int m, int p, int n) {
	float time = 0.f;
	float* a_gpu = nullptr, *b_gpu = nullptr, *c_gpu = nullptr;
	cudaMalloc((void**)&a_gpu, m * p * sizeof(float));
	cudaMalloc((void**)&b_gpu, p * n * sizeof(float));
	cudaMalloc((void**)&c_gpu, m * n * sizeof(float));
	cudaMemcpy(a_gpu, a, m * p * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, p * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_gpu, c, m * n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block_size_3d(MATRIX_BLOCK_SIZE, MATRIX_BLOCK_SIZE), num_blocks_3d;
	num_blocks_3d.x = (m * n + block_size_3d.x - 1) / block_size_3d.x;
	num_blocks_3d.y = (m * n + block_size_3d.y - 1) / block_size_3d.y;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	block_multiply <<< num_blocks_3d, block_size_3d >>> (a_gpu, b_gpu, c_gpu, m, p, n);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaMemcpy(c, c_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(c_gpu);
	return time / 1000;
}
