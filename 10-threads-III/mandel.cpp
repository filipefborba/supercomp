#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NPOINTS 1000
#define MAXITER 10000

struct d_complex {
   double r;
   double i;
};

int testpoint(d_complex c) {
   struct d_complex z;
   double temp;
   z = c;
   for (int iter = 0; iter < MAXITER; iter++) {
      temp = (z.r * z.r) - (z.i * z.i) + c.r;
      z.i = z.r * z.i * 2 + c.i;
      z.r = temp;
      if ((z.r * z.r + z.i * z.i) > 4.0)
      {
         return 1;
      }
   }
   return 0;
}

int main() {
   struct d_complex c;
   int numoutside = 0;
   double area, error, eps = 1.0e-5;

   #pragma omp parallel for reduction(+:numoutside) private(c)
      for (int i = 0; i < NPOINTS; i++) {
         for (int j = 0; j < NPOINTS; j++) {
            c.r = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + eps;
            c.i = 1.125 * (double)(j) / (double)(NPOINTS) + eps;
            numoutside += testpoint(c);
         }
      }

   area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
   error = area / (double)NPOINTS;
   printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
   printf("Correct answer should be around 1.510659\n");
   printf("numoutside: %d\n", numoutside);
}
