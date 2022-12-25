#include <assert.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

__global__ void kernel(const int* arr, int* result, unsigned arr_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%dth block. %dth thread (global index: %d)\n", blockIdx.x, threadIdx.x, idx);
    if (idx < arr_size)
    {
        result[idx] = arr[idx] + idx;
    }
}

int main()
{
    cudaError_t err = cudaSuccess;
    const unsigned VECT_SIZE = 100, BLOCK_SIZE = 4;
    unsigned num_blocks = (VECT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Blocks: %u. Block size: %u \n", num_blocks, BLOCK_SIZE);
    int *vect_gpu = nullptr, *vect = new int[VECT_SIZE];
    int *result_gpu = nullptr, *result = new int[VECT_SIZE];
    for (size_t i = 0; i < VECT_SIZE; i++)
    {
        vect[i] = 1;
        result[i] = 0;
    }
    cudaMalloc((void**)&vect_gpu, VECT_SIZE * sizeof(int));
    cudaMalloc((void**)&result_gpu, VECT_SIZE * sizeof(int));
    cudaMemcpy(vect_gpu, vect, VECT_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block_size_3d(BLOCK_SIZE), num_blocks_3d(num_blocks);
    kernel <<<num_blocks_3d, block_size_3d>>>(vect_gpu, result_gpu, VECT_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(result, result_gpu, VECT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("(");
    for (size_t i = 0; i < VECT_SIZE; i++)
    {
        printf("%d; ", result[i]);
    }
    printf(")\n");
    delete[] vect;
    delete[] result;
    cudaFree(vect_gpu);
    cudaFree(result_gpu);
}
