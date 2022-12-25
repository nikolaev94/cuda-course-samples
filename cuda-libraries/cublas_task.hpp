#pragma once

#include <random>

#include <cuda_runtime_api.h>
#include <cublas.h>

const int BLAS_MAX_ITERS = 10000;
const int BLAS_SYSTEM_SIZE = 8;
const float BLAS_MIN = -2.f, BLAS_MAX = 2.f;

void init_minimal_residual(float* a, float* b, float* x_cpu, float* x_gpu, int n);

void minimal_residual_gpu(float* a, float* b, float* x, int n);

void minimal_residual_cpu(float* a, float* b, float* x, int n);

float get_minimal_residual_error(int size, const float* a, const float* x_vec, const float* b_vec);

// y_vec[] := alpha * A[] * x_vec[] + beta * y_vec[]
void sgemv_cpu(int size, float alpha, const float* A, const float* x_vec, float beta, float* y_vec);

// y_vec[] := alpha * x_vec[] + y_vec[]
void saxpy_cpu(int size, float alpha, const float* x_vec, float* y_vec);

// dot := (x_vec[], y_vec[])
float sdot_cpu(int size, const float* x_vec, const float* y_vec);

float euclidian_norm(int size, const float* x_vec);
