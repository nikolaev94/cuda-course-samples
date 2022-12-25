#include "cublas_task.hpp"

void init_minimal_residual(float* a, float* b, float* x_cpu, float* x_gpu, int n) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr_sp(BLAS_MIN, BLAS_MAX);
    for (int i = 0; i < n; i++) {
        x_cpu[i] = x_gpu[i] = distr_sp(generator);
        b[i] = distr_sp(generator);
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (j != i) {
                a[j*n + i] = distr_sp(generator);
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        float sum = 0.f;
        for (int j = 0; j < n; j++)
        {
            if (i != j)
            {
                sum += std::fabs(a[j*n + i]);
            }
        }
        if (rand() % 2 == 0) {
            a[i*n + i] = sum + 1.f;
        }
        else {
            a[i*n + i] = -sum - 1.f;
        }
    }
}

void minimal_residual_gpu(float* a, float* b, float* x, int n) {
    float* a_gpu = nullptr, *b_gpu = nullptr, *x_gpu = nullptr;
    float* r = nullptr, *ar = nullptr;
    float tau = 0.f;
    cublasInit();
    cublasAlloc(n * n, sizeof(float), (void**)&a_gpu);
    cublasAlloc(n, sizeof(float), (void**)&b_gpu);
    cublasAlloc(n, sizeof(float), (void**)&x_gpu);
    cublasAlloc(n, sizeof(float), (void**)&r);
    cublasAlloc(n, sizeof(float), (void**)&ar);
    cublasSetVector(n, sizeof(float), b, 1, b_gpu, 1);
    cublasSetVector(n, sizeof(float), x, 1, x_gpu, 1);
    cublasSetVector(n * n, sizeof(float), a, 1, a_gpu, 1);
    unsigned iteration = 0;
    while (iteration < BLAS_MAX_ITERS) {
        cublasScopy(n, b_gpu, 1, r, 1);
        // r := A * x - r
        cublasSgemv('t', n, n, 1.f, a_gpu, n, x_gpu, 1, -1.f, r, 1);
        // Ar := A * r
        cublasSgemv('t', n, n, 1.f, a_gpu, n, r, 1, 0.f, ar, 1);
        // tau := (Ar, r) / (Ar, Ar)
        tau = cublasSdot(n, ar, 1, r, 1) / cublasSdot(n, ar, 1, ar, 1);
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

void minimal_residual_cpu(float* a, float* b, float* x, int n)
{
    float* r = new float[n], *ar = new float[n];
    memset(r, 0, n * sizeof(float));
    memset(ar, 0, n * sizeof(float));
    float tau = 0.f;
    unsigned iteration = 0;
    while (iteration < BLAS_MAX_ITERS)
    {
        memcpy(r, b, n * sizeof(float));
        // r := A * x - b
        sgemv_cpu(n, 1.f, a, x, -1.f, r);
        // Ar := A * r
        sgemv_cpu(n, 1.f, a, r, 0.f, ar);
        // tau := (Ar, r) / (Ar, Ar)
        tau = sdot_cpu(n, ar, r) / sdot_cpu(n, ar, ar);
        // x := -tau * r + x
        saxpy_cpu(n, -1.f * tau, r, x);
        ++iteration;
    }
    delete[] r;
    delete[] ar;
}

// y_vec[] := alpha * A[] * x_vec[] + beta * y_vec[]
void sgemv_cpu(int size, float alpha, const float* A, const float* x_vec, float beta, float* y_vec)
{
    for (size_t j = 0; j < size; ++j)
    {
        float sum = 0.f;
        for (size_t i = 0; i < size; ++i)
        {
            sum += alpha * A[j * size + i] * x_vec[i];
        }
        y_vec[j] = sum + beta * y_vec[j];
    }
}

// y_vec[] := alpha * x_vec[] + y_vec[]
void saxpy_cpu(int size, float alpha, const float* x_vec, float* y_vec)
{
    for (size_t i = 0; i < size; ++i)
    {
        y_vec[i] += alpha * x_vec[i];
    }
}

// dot := (x_vec[], y_vec[])
float sdot_cpu(int size, const float* x_vec, const float* y_vec)
{
    float product = 0.f;
    for (size_t i = 0; i < size; ++i)
    {
        product += x_vec[i] * y_vec[i];
    }
    return product;
}

// A * x = b
float get_minimal_residual_error(int size, const float* A, const float* x_vec, const float* b_vec)
{
    float *ax_vec = new float[size];
    memset(ax_vec, 0, size * sizeof(float));
    sgemv_cpu(size, 1.f, A, x_vec, 0.f, ax_vec);
    for (size_t i = 0; i < size; ++i)
    {
        ax_vec[i] = ax_vec[i] - b_vec[i];
    }
    float error = euclidian_norm(size, ax_vec);
    delete[] ax_vec;
    return error;
}

float euclidian_norm(int size, const float* x_vec)
{
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i)
    {
        sum += x_vec[i] * x_vec[i];
    }
    return std::sqrtf(sum);
}
