#include <ctime>
#include <iostream>
#include <random>

#include <omp.h>

#include "axpy.hpp"

void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy)
{
    omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        if (((i * incy) < n) && ((i * incx) < n))
        {
            y[i * incy] = y[i * incy] + a * x[i * incx];
        }
    }
}

void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy)
{
	omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		if (((i * incy) < n) && ((i * incx) < n))
		{
			y[i * incy] = y[i * incy] + a * x[i * incx];
		}
	}
}

void main()
{
    const unsigned VECT_SIZE = 1'000'000U;
    double* dp_x_vect = new double[VECT_SIZE];
	double* dp_y_vect_cpu = new double[VECT_SIZE];
	double* dp_y_vect_gpu = new double[VECT_SIZE];
    float* sp_x_vect = new float[VECT_SIZE];
    float* sp_y_vect_cpu = new float[VECT_SIZE];
    float* sp_y_vect_gpu = new float[VECT_SIZE];

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr_fl(-8.0, 8.0);
    std::uniform_real_distribution<double> distr(-8.0, 8.0);
    double dp_a_param = distr(generator);
    float sp_a_param = distr_fl(generator);
    for (size_t i = 0; i < VECT_SIZE; i++)
    {
        dp_x_vect[i] = distr(generator);
        sp_x_vect[i] = distr_fl(generator);
        dp_y_vect_cpu[i] = dp_y_vect_gpu[i] = distr(generator);
        sp_y_vect_cpu[i] = sp_y_vect_gpu[i] = distr_fl(generator);
    }

    clock_t start = 0L;
    float time_sec = 0.f;

	start = clock();
    saxpy_omp(VECT_SIZE, sp_a_param, sp_x_vect, 1, sp_y_vect_cpu, 1);
	time_sec = (float)(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "Kernel time (CPU)(float): " << time_sec << std::endl;

    start = clock();
    daxpy_omp(VECT_SIZE, dp_a_param, dp_x_vect, 1, dp_y_vect_cpu, 1);
    time_sec = (float)(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "Kernel time (CPU)(double): " << time_sec << std::endl;

	start = clock();
    saxpy_gpu(VECT_SIZE, sp_a_param, sp_x_vect, 1, sp_y_vect_gpu, 1);
	time_sec = (float)(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "Kernel time (GPU)(float): " << time_sec << std::endl;

    start = clock();
    daxpy_gpu(VECT_SIZE, dp_a_param, dp_x_vect, 1, dp_y_vect_gpu, 1);
    time_sec = (float)(clock() - start) / CLOCKS_PER_SEC;
    std::cout << "Kernel time (GPU)(double): " << time_sec << std::endl;

    double max_error = std::numeric_limits<float>::epsilon();
    float max_error_fl = std::numeric_limits<float>::epsilon();
    for (size_t i = 0; i < VECT_SIZE; i++)
    {
        float error_fl = std::fabs(sp_y_vect_cpu[i] - sp_y_vect_gpu[i]);
        if (error_fl > max_error_fl)
        {
            max_error_fl = error_fl;
        }
        double error = std::fabs(dp_y_vect_cpu[i] - dp_y_vect_gpu[i]);
        if (error > max_error)
        {
            max_error = error;
        }
    }
    std::cout << "Max error (CPU vs. GPU)(float): " << max_error_fl << std::endl;
    std::cout << "Max error (CPU vs. GPU)(double): " << max_error << std::endl;
    delete[] sp_y_vect_cpu;
    delete[] sp_y_vect_gpu;
    delete[] sp_x_vect;

	delete[] dp_y_vect_gpu;
	delete[] dp_y_vect_cpu;
	delete[] dp_x_vect;
}
