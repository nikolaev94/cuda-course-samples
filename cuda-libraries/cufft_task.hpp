#pragma once

#include <random>

#include <cuda_runtime_api.h>
#include <cufft.h>

const unsigned FFT_BLOCK_SIZE = 16;

const int FFT_WAVE_SIZE = 10000;
const float FFT_MIN = -10.f, FFT_MAX = 10.f;

typedef cufftComplex complex;

void init_fft(complex* a, complex* b, complex* c_cpu, complex* c_gpu, int n);

void fft_cpu(const complex* a, const complex* b, complex* c, int n);

void fft_gpu(const complex* a, const complex* b, complex* c, int n);
