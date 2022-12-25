#ifndef AXPY_H
#define AXPY_H

const int BLOCK_SIZE = 256;

void saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy);

void daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy);

#endif // AXPY_H
