#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

__global__ void calc_pi(double *dev, double step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double x = (i + 0.5) * step;
    dev[i] = 4.0 /(1.0 + x * x);
}

int main() {
    static long num_steps = 1000000000;
    static int gpu_threads = 1024;
    double step;
    double pi, sum = 0.0;
    step = 1.0 / (double) num_steps;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    

    thrust::device_vector<double> dev(num_steps);
    calc_pi<<<ceil((double) num_steps/gpu_threads), gpu_threads>>>(thrust::raw_pointer_cast(dev.data()), step);
    sum = thrust::reduce(dev.begin(), dev.end(), (double) 0, thrust::plus<double>());
    pi = step * sum;

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    
    printf("O valor de pi calculado com %ld passos levou \n", num_steps);
    printf("%.2f milisegundo(s) e chegou no valor: \n", msecTotal);
    printf("%.17f\n", pi);
}