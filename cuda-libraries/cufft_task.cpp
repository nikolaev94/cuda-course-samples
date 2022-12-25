#include "cufft_task.hpp"

void init_fft(complex* a, complex* b, complex* c_cpu, complex* c_gpu, int n)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distr_sp(FFT_MIN, FFT_MAX);
    for (int i = 0; i < n; i++) {
        a[i] = { distr_sp(generator), distr_sp(generator) };
        b[i] = { distr_sp(generator), distr_sp(generator) };
        c_cpu[i] = c_gpu[i] = { 0.f, 0.f };
    }
}

void fft_cpu(const complex* a, const complex* b, complex* c, int n) {
    for (int i = 0; i < n; i++)
    {
        float sx = 0;
        float sy = 0;
        for (int j = 0; j < n; j++)
        {
            int k = i - j;
            if (k < 0)
            {
                k += n;
            }
            sx += a[j].x * b[k].x - a[j].y * b[k].y;
            sy += a[j].x * b[k].y + a[j].y * b[k].x;
        }
        c[i] = { sx, sy };
    }
}
