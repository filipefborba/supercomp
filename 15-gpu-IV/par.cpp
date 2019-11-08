// Código Matrix multiplication: C = A * B. Super simplificado.

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>

#define BLOCK_SIZE 32
#ifndef VECTOR_SIZE
    #define VECTOR_SIZE 128
#endif

void MatrixMulCpuPar(float* M, float* N, float* P, int size) {
    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            for (k = 0; k < size; k++) {
                P[i*size+j] += M[i*size+k] * N[k*size+j];
            }
        }
    }
}

int main(int argc, char **argv)
{

    const int size = VECTOR_SIZE;
    // const int nIter = 100000;

    float *h_A = (float *)malloc(sizeof(float) * size * size);
    float *h_B = (float *)malloc(sizeof(float) * size * size);
    float *h_C = (float *)malloc(sizeof(float) * size * size);
    for (int i = 0; i < size * size; ++i) { h_A[i] =  1.0f; h_B[i] =  1.0f; }

    // Nao precisa alocar GPU
    // float *d_A, *d_B, *d_C;
    // cudaMalloc((void **) &d_A, sizeof(float) * size * size);
    // cudaMalloc((void **) &d_B, sizeof(float) * size * size);
    // cudaMalloc((void **) &d_C, sizeof(float) * size * size);

    // cudaMemcpy(d_A, h_A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    // dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid(size / threads.x, size / threads.y);

    // PARALELO
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    // for (int j = 0; j < nIter; j++) {
        MatrixMulCpuPar(h_A, h_B, h_C, size);
    // }
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    float msecPerMatrixMul = msecTotal;
    printf("Time= %f\n", msecPerMatrixMul);
    // Copy result from device to host
    // cudaMemcpy(h_C, d_C, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(size * size); i++)
    {
        double abs_err = fabs(h_C[i] - (size * 1.0f));
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/size ;
        if (rel_err > eps)
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], size*1.0f, eps);
    }

    free(h_A); free(h_B); free(h_C);
    // cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return(0);
}
