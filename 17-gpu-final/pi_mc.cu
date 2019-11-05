#include <stdio.h>
#include "curand.h"
#include "curand_kernel.h"
#include "math.h"
#include <thrust/device_vector.h>

__global__ void calc_pi(int *dev, long num_trials, double r) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx >= num_trials) return;

   double x, y, test;
   int Ncirc = 0;

   curandState st;
   curand_init(0, idx, 0, &st);

   for (int i = 0; i < 4192; i++)
   {
      x = curand_uniform(&st);
      y = curand_uniform(&st);
      test = x * x + y * y;
      if (test <= r * r) {
         Ncirc++;
      }
   }
   dev[idx] = Ncirc;
}


int main()
{
   static long num_trials = 1000000000;
   static int gpu_threads = 1024;
   static long nblocks = ceil(num_trials / (gpu_threads * 4192) );
   double r = 1.0; // radius of circle. Side of squrare is 2*r

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, NULL);

   thrust::device_vector<int> dev(nblocks*gpu_threads);
   calc_pi<<<nblocks, gpu_threads>>>(thrust::raw_pointer_cast(dev.data()), num_trials, r);
   double Ncirc = thrust::reduce(dev.begin(), dev.end(), 0.0, thrust::plus<double>());
   double pi = 4.0 * ((double) Ncirc / (double) num_trials);   

   cudaEventRecord(stop, NULL);
   cudaEventSynchronize(stop);
   float msecTotal = 0.0f;
   cudaEventElapsedTime(&msecTotal, start, stop);

   printf("\n%ld trials, pi is %lf \n", num_trials, pi);
   printf("%.2f milisegundo(s). \n", msecTotal);

   return 0;
}
