// Código Matrix multiplication: C = A * B. Super simplificado.

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 32
#ifndef VECTOR_SIZE
    #define VECTOR_SIZE 128
#endif

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) 
{
    // Calcule o índice da linha do elemento em P e M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;

    // Calcule o índice da coluna do elemento em P e N
    int Col = blockIdx.x*blockDim.x+threadIdx.x;

    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0;
        // cada thread calcula um elemento da sub-matriz

        for (int k = 0; k < Width; ++k) {
        Pvalue += M[Row*Width+k]*N[k*Width+Col];
        }

        P[Row*Width+Col] = Pvalue;
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

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(float) * size * size);
    cudaMalloc((void **) &d_B, sizeof(float) * size * size);
    cudaMalloc((void **) &d_C, sizeof(float) * size * size);

    cudaMemcpy(d_A, h_A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * size * size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(size / threads.x, size / threads.y);
    
    // -------------------------------------------------------------------------------
    // MATRIXMULKERNEL
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    // for (int j = 0; j < nIter; j++)
    MatrixMulKernel<<<grid,threads>>>(d_A, d_B, d_C, size);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    // float msecPerMatrixMul = msecTotal / nIter;
    float msecPerMatrixMul = msecTotal;
    printf("Time= %f\n", msecPerMatrixMul);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

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
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return(0);
}
