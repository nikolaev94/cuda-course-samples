#include <iostream>

#include "cublas_task.hpp"
#include "cufft_task.hpp"
#include "curand_task.hpp"

void minimal_residual()
{
    std::cout << "Minimal residual. Linear system size: " << BLAS_SYSTEM_SIZE << 'x' << BLAS_SYSTEM_SIZE << std::endl;
    std::cout << "Minimal residual. Method iterations: " << BLAS_MAX_ITERS << std::endl;
    float* a_matr = new float[BLAS_SYSTEM_SIZE * BLAS_SYSTEM_SIZE];
    float* b_vect = new float[BLAS_SYSTEM_SIZE];
    float* x_vect_cpu = new float[BLAS_SYSTEM_SIZE];
    float* x_vect_gpu = new float[BLAS_SYSTEM_SIZE];
    init_minimal_residual(a_matr, b_vect, x_vect_cpu, x_vect_gpu, BLAS_SYSTEM_SIZE);
    minimal_residual_cpu(a_matr, b_vect, x_vect_cpu, BLAS_SYSTEM_SIZE);
    auto cpu_error = get_minimal_residual_error(BLAS_SYSTEM_SIZE, a_matr, x_vect_cpu, b_vect);
    std::cout << "Minimal residual. A*x - b error (CPU): " << cpu_error << std::endl;
    minimal_residual_gpu(a_matr, b_vect, x_vect_gpu, BLAS_SYSTEM_SIZE);
    auto gpu_error = get_minimal_residual_error(BLAS_SYSTEM_SIZE, a_matr, x_vect_gpu, b_vect);
    std::cout << "Minimal residual. A*x - b error (GPU): " << gpu_error << std::endl;

    delete[] a_matr;
    delete[] b_vect;
    delete[] x_vect_cpu;
    delete[] x_vect_gpu;
}

void fft_convolution()
{
    std::cout << "FFT convolution. Wave size: " << FFT_WAVE_SIZE << std::endl;
    complex* a_wave = new complex[FFT_WAVE_SIZE];
    complex* b_wave = new complex[FFT_WAVE_SIZE];
    complex* c_wave_cpu = new complex[FFT_WAVE_SIZE];
    complex* c_wave_gpu = new complex[FFT_WAVE_SIZE];
    init_fft(a_wave, b_wave, c_wave_cpu, c_wave_gpu, FFT_WAVE_SIZE);
    fft_cpu(a_wave, b_wave, c_wave_cpu, FFT_WAVE_SIZE);
    fft_gpu(a_wave, b_wave, c_wave_gpu, FFT_WAVE_SIZE);
    float max_diff = -1.f;
    for (size_t i = 0; i < FFT_WAVE_SIZE; ++i)
    {
        complex comp_diff = { c_wave_cpu[i].x - c_wave_gpu[i].x, c_wave_cpu[i].y - c_wave_cpu[i].y };
        float diff = std::sqrtf(comp_diff.x * comp_diff.x + comp_diff.y * comp_diff.y);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    std::cout << "FFT convolution. Max error (CPU vs. GPU): " << max_diff << std::endl;
    delete[] a_wave;
    delete[] b_wave;
    delete[] c_wave_cpu;
    delete[] c_wave_gpu;
}

void montecarlo_integral()
{
    std::cout << "Montecarlo integral. Number of points: " << MCARLO_POINTS << std::endl;
    std::cout << "Montecarlo integral. a = " << MCARLO_A << " b = " << MCARLO_B << std::endl;
    unsigned* counts_host_api = new unsigned[MCARLO_POINTS];
    unsigned* counts_device_api = new unsigned[MCARLO_POINTS];
    memset(counts_host_api, 0, MCARLO_POINTS * sizeof(unsigned));
    memset(counts_device_api, 0, MCARLO_POINTS * sizeof(unsigned));
    montecarlo_gpu_host_api(MCARLO_A, MCARLO_B, counts_host_api, MCARLO_POINTS);
    montecarlo_gpu_device_api(MCARLO_A, MCARLO_B, counts_device_api, MCARLO_POINTS);
    unsigned count_host_api = 0, count_device_api = 0;
    for (int i = 0; i < MCARLO_POINTS; i++) {
        count_host_api += counts_host_api[i];
        count_device_api += counts_device_api[i];
    }
    delete[] counts_host_api;
    delete[] counts_device_api;
    float reference = std::fabsf(std::atanf(MCARLO_B * 1.f) - std::atanf(MCARLO_A * 1.f));
    float measured_host_api = 1.f * (MCARLO_B - MCARLO_A) * count_host_api / MCARLO_POINTS;
    float measured_device_api = 1.f * (MCARLO_B - MCARLO_A) * count_device_api / MCARLO_POINTS;
    float diff_host_api = std::fabsf(measured_host_api - reference);
    float diff_device_api = std::fabsf(measured_device_api - reference);
    std::cout << "Montecarlo integral. Reference: " << reference << std::endl;
    std::cout << "Montecarlo integral. Measured (Host API): " << measured_host_api << std::endl;
    std::cout << "Montecarlo integral. Measured (Device API): " << measured_device_api << std::endl;
    std::cout << "Montecarlo integral. Error (CPU vs. Host API): " << diff_host_api << std::endl;
    std::cout << "Montecarlo integral. Error (CPU vs. Device API): " << diff_device_api << std::endl;
}

int main()
{
    minimal_residual();

    fft_convolution();

    montecarlo_integral();

    return 0;
}
