#include <ctime>
#include <iostream>
#include <random>

#include "gemm.hpp"

void matrix_multiply_cpu(float* a, float* b, float* c_cpu, int M_SIZE, int P_SIZE, int N_SIZE) {
	// m*p p*n => m*n
	for (int i = 0; i < M_SIZE; i++) {
		for (int j = 0; j < N_SIZE; j++) {
			float s = 0;
			for (int k = 0; k < P_SIZE; k++) {
				s += a[i*P_SIZE + k] * b[k*N_SIZE + j];
			}
			c_cpu[i*N_SIZE + j] = s;
		}
	}
}

void main() {
	const int M_SIZE = 256, P_SIZE = 256, N_SIZE = 256;
	float* a = new float[M_SIZE * P_SIZE];
	float* b = new float[P_SIZE * N_SIZE];
	float* c_cpu = new float[M_SIZE * N_SIZE];
	float* c_gpu_naive = new float[M_SIZE * N_SIZE];
	float* c_gpu_block = new float[M_SIZE * N_SIZE];

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr_fl(-10.f, 10.f);
	for (int i = 0; i < M_SIZE * P_SIZE; i++) {
		a[i] = distr_fl(generator);
	}
	for (int i = 0; i < P_SIZE * N_SIZE; i++) {
		b[i] = distr_fl(generator);;
	}
	for (int i = 0; i < M_SIZE * N_SIZE; i++) {
		c_cpu[i] = 0.f;
		c_gpu_naive[i] = 0.f;
		c_gpu_block[i] = 0.f;
	}

	clock_t start = clock();
	matrix_multiply_cpu(a, b, c_cpu, M_SIZE, P_SIZE, N_SIZE);
	float time_cpu_sec = float(clock() - start) / CLOCKS_PER_SEC;

	float time_naive_gpu_sec(0.f), time_block_gpu_sec(0.f);
	time_naive_gpu_sec = naive_multiply_gpu(a, b, c_gpu_naive, M_SIZE, P_SIZE, N_SIZE);
	time_block_gpu_sec = block_multiply_gpu(a, b, c_gpu_block, M_SIZE, P_SIZE, N_SIZE);

    std::cout << "Matrix: " << M_SIZE << '*' << P_SIZE << '*' << N_SIZE << std::endl;
    std::cout << "CPU time (sec): " << time_cpu_sec << std::endl;
    std::cout << "GPU naive time (sec): " << time_naive_gpu_sec << std::endl;
    std::cout << "GPU block time (sec): " << time_block_gpu_sec << std::endl;
	float max_diff_naive = -1.f, max_diff_block = -1.f;
	for (int i = 0; i < M_SIZE * N_SIZE; i++) {
        float diff_naive = std::fabs(c_gpu_naive[i] - c_cpu[i]);
        float diff_block = std::fabs(c_gpu_block[i] - c_cpu[i]);
		if (diff_naive > max_diff_naive) {
			max_diff_naive = diff_naive;
		}
		if (diff_block > max_diff_block) {
			max_diff_block = diff_block;
		}
	}
    std::cout << "Measurement error naive: " << max_diff_naive << std::endl;
    std::cout << "Measurement error block: " << max_diff_block << std::endl;
	delete[] a;
	delete[] b;
	delete[] c_cpu;
	delete[] c_gpu_naive;
	delete[] c_gpu_block;
}
