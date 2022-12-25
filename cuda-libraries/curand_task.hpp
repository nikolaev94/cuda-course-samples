#pragma once

const unsigned MCARLO_BLOCK_SIZE = 16;

const unsigned MCARLO_POINTS = 10000;
const int MCARLO_A = 0, MCARLO_B = 100;

void montecarlo_gpu_host_api(float a, float b, unsigned* counts, int n);

void montecarlo_gpu_device_api(float a, float b, unsigned* counts, int n);
