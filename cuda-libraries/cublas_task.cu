#include <cuda_runtime_api.h>
#include <cublas.h>

#include<iostream>

#include "cuda_libraries.h"

void minimal_residual_gpu(float* a, float* b, float* x, int n) {
    float* a_gpu = nullptr, *b_gpu = nullptr, *x_gpu = nullptr;
    float* r = nullptr, *ar = nullptr;
    float tau = 0.0f;
    cublasInit();
    cublasAlloc(n * n, sizeof(float), (void**)&a_gpu);
    cublasAlloc(n, sizeof(float), (void**)&b_gpu);
    cublasAlloc(n, sizeof(float), (void**)&x_gpu);
    cublasAlloc(n, sizeof(float), (void**)&r);
    cublasAlloc(n, sizeof(float), (void**)&ar);
    cublasSetVector(n, sizeof(float), b, 1, b_gpu, 1);
    cublasSetVector(n, sizeof(float), x, 1, x_gpu, 1);
    cublasSetVector(n * n, sizeof(float), a, 1, a_gpu, 1);
    cublasScopy(n, b_gpu, 1, r, 1);
    int iteration = 0;
    std::cout << std::endl << std::endl;
    while (iteration < 100) {
        // r := A * x - r
        cublasSgemv('N', n, n, 1.f, a_gpu, n, x_gpu, 1, -1.f, r, 1);
        // Ar := A * r
        cublasSgemv('N', n, n, 1.f, a_gpu, n, r, 1, 0.f, ar, 1);
        // tau := (Ar, r) / (Ar, Ar)
        tau = cublasSdot(n, ar, 1, r, 1) / cublasSdot(n, ar, 1, ar, 1);
        std::cout << tau << std::endl;
        // x := -tau * r + x
        cublasSaxpy(n, -1.f * tau, r, 1, x_gpu, 1);
        iteration++;
    }
    cublasGetVector(n, sizeof(float), x_gpu, 1, x, 1);
    cublasFree(a_gpu);
    cublasFree(b_gpu);
    cublasFree(x_gpu);
    cublasFree(r);
    cublasFree(ar);
    cublasShutdown();
}
