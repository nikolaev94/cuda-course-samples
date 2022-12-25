#pragma once

float naive_multiply_gpu(float* a, float* b, float* c, int m, int p, int n);

float block_multiply_gpu(float* a, float* b, float* c, int m, int p, int n);
