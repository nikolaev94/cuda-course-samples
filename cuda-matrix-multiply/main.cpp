#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>

#include "matrix_multiply.hpp"

void init(float* a, float* b, float* c_cpu, float* c_gpu, float& alpha, int n) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr_fl(-10.f, 10.f);
    alpha = distr_fl(generator);
	for (int i = 0; i < n * n; i++) {
        a[i] = distr_fl(generator);
        b[i] = distr_fl(generator);
		c_cpu[i] = c_gpu[i] = 0.f;
	}
}

void task_cpu(const float* a, const float* b, float* c, float alpha, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i * n + j] = a[i * n + j] + alpha * b[j * n + i];
		}
	}
}

void main() {
	const int N_SIZE = 300; // matrix size
	float alpha = -1.0;
	float* a = new float[N_SIZE * N_SIZE];
	float* b = new float[N_SIZE * N_SIZE];
	float* c_cpu = new float[N_SIZE * N_SIZE];
	float* c_gpu = new float[N_SIZE * N_SIZE];
	init(a, b, c_cpu, c_gpu, alpha, N_SIZE);
	task_cpu(a, b, c_cpu, alpha, N_SIZE);
	task_gpu(a, b, c_gpu, alpha, N_SIZE);

	float max_diff = -1.f;
	for (int i = 0; i < N_SIZE * N_SIZE; i++) {
		float diff = std::fabs(c_gpu[i] - c_cpu[i]);
		if (diff > max_diff) {
			max_diff = diff;
		}
	}
    std::cout << "Max error (CPU vs.GPU): " << max_diff << std::endl;
	delete[] a;
	delete[] b;
	delete[] c_cpu;
	delete[] c_gpu;
}
