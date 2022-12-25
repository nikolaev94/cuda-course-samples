#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include "curand_task.hpp"

typedef curandStateXORWOW_t randState;

__device__ float test_function(float x)
{
	return (1 / (1 + x * x));
}

__global__ void check_points(float a, float b, unsigned* counts, int n, float* x, float* y)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
    {
        float x_norm = a + (b - a) * x[idx];
		if (y[idx] >= test_function(x_norm)) {
			counts[idx] = 0;
		} else {
			counts[idx] = 1;
		}
	}
}

__global__ void setup_random_state(randState* states, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void generate_check_points(randState* states, float a, float b, unsigned* counts, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float x_norm = a + (b - a) * curand_uniform(&states[idx]);
        float y_gen = curand_uniform(&states[idx]);
        if (y_gen >= test_function(x_norm)) {
            counts[idx] = 0;
        }
        else {
            counts[idx] = 1;
        }
    }
}

void montecarlo_gpu_host_api(float a, float b, unsigned* counts , int n)
{
	float* x_gpu = nullptr, *y_gpu = nullptr;
    unsigned* counts_gpu = nullptr;
	cudaMalloc((void**)&x_gpu, sizeof(float) * n);
	cudaMalloc((void**)&y_gpu, sizeof(float) * n);
	cudaMalloc((void**)&counts_gpu, sizeof(unsigned) * n);
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	curandGenerateUniform(generator, x_gpu, n);
	curandGenerateUniform(generator, y_gpu, n);
	curandDestroyGenerator(generator);
	unsigned num_blocks = (n + MCARLO_BLOCK_SIZE - 1) / MCARLO_BLOCK_SIZE;
	check_points <<< num_blocks, MCARLO_BLOCK_SIZE >>>(a, b, counts_gpu, n, x_gpu, y_gpu);
	cudaMemcpy(counts, counts_gpu, sizeof(unsigned) * n, cudaMemcpyDeviceToHost);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(counts_gpu);
}

void montecarlo_gpu_device_api(float a, float b, unsigned* counts, int n) {
	unsigned *counts_gpu = nullptr;
    curandStateXORWOW_t* dev_states = nullptr;
    unsigned num_blocks = (n + MCARLO_BLOCK_SIZE - 1) / MCARLO_BLOCK_SIZE;
	cudaMalloc((void**)&counts_gpu, sizeof(unsigned) * n);
	cudaMalloc((void**)&dev_states, num_blocks * MCARLO_BLOCK_SIZE * sizeof(curandStateXORWOW_t));
    setup_random_state <<< num_blocks, MCARLO_BLOCK_SIZE >>> (dev_states, time(nullptr));
    generate_check_points <<< num_blocks, MCARLO_BLOCK_SIZE >>> (dev_states, a, b, counts_gpu, n);
	cudaMemcpy(counts, counts_gpu, sizeof(unsigned) * n, cudaMemcpyDeviceToHost);
	cudaFree(counts_gpu);
	cudaFree(dev_states);
}
