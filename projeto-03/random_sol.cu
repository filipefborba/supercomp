#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include "curand.h"
#include "curand_kernel.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdio.h>

#define BLOCK_SIZE 32

__global__ void calc_dist(double* X, double* Y, double* Dist, int N) {
    int i = blockIdx.y*blockDim.y+threadIdx.y;
    int j = blockIdx.x*blockDim.x+threadIdx.x;

    // Essa matriz é simetrica, mas estamos calculando ela inteira. Ponto de otimizacao!
    Dist[i*N+j] = sqrt(pow((X[i] - X[j]), 2) + pow((Y[i] - Y[j]), 2));
}

__device__ void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    return;
}

__global__ void random_sol(int *solutions, double *costs, double *distances, int N) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    double solution_cost = 0; // Custo total dessa solucao

    // Preenche a solucao em ordem para que possamos permutar depois
    for (int k = 0; k < N; k++) {
        solutions[i * N + k] = k;
    }
    
    // Inicializar o random
    curandState_t st;
    curand_init(0, i, 0, &st);
    int idx;

    // Realiza a permutacao e calcula o custo total da solucao
    for (int k = 1; k < N; k++){
        idx = (int) ((N-k) * curand_uniform(&st) + k); // Pegar um indice aleatorio entre 1 e N-1

        // Swap dos elementos do vetor e salva no vetor de solucoes
        swap(&solutions[i * N + k], &solutions[i * N + idx]);

        solution_cost += distances[solutions[i * N + k-1] * N + solutions[i * N + k]]; // Calculo das distancias
    }
    solution_cost += distances[solutions[i * N] * N + solutions[i * N + N-1]]; // Ultimo calculo: primeiro e ultimo

    costs[i] = solution_cost; // Salva no vetor de custos
}

int main() {
    // Preparacao para receber os dados do arquivo
    int N; std::cin >> N;
    thrust::host_vector<double> host_x(N);
    thrust::host_vector<double> host_y(N);

    double x, y;
    for (int i = 0; i < N; i++) {
        std::cin >> x; std::cin >> y;
        host_x[i] = x;
        host_y[i] = y;
    }
    // ---------------------------------------------------------------------
    // Preparacao para pre-calcular as distancias
    thrust::device_vector<double> dev_x(host_x);
    thrust::device_vector<double> dev_y(host_y);
    thrust::device_vector<double> dev_points_distance(N * N);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((double) N / threads.x), ceil((double) N / threads.y));

    calc_dist<<<grid,threads>>>(thrust::raw_pointer_cast(dev_x.data()), 
        thrust::raw_pointer_cast(dev_y.data()),
        thrust::raw_pointer_cast(dev_points_distance.data()), 
        N);

    // ---------------------------------------------------------------------
    // Preparacao sortear solucoes e calcular custos
    long nSols = 10000;
    int gpu_threads = 1024;
    
    thrust::device_vector<int> dev_solutions(nSols * N); // Vetor de solucoes
    thrust::device_vector<double> dev_costs(nSols); // Vetor de custos totais de cada solucao

    // Medicao de Tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    
    random_sol<<<ceil((double) nSols/gpu_threads), gpu_threads>>>(thrust::raw_pointer_cast(dev_solutions.data()), 
        thrust::raw_pointer_cast(dev_costs.data()), 
        thrust::raw_pointer_cast(dev_points_distance.data()), 
        N);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // ---------------------------------------------------------------------
    // Pegar o elemento minimo do vetor
    thrust::device_vector<double>::iterator iter = thrust::min_element(dev_costs.begin(), dev_costs.end());
    int position = iter - dev_costs.begin();
    double min_val = *iter;

    // ---------------------------------------------------------------------
    // Print do tempo e do melhor caminho
    #ifdef TIME
        std::cout << msecTotal << std::endl;
        std::cout << "milisegundo(s)." << std::endl;
    #endif

    std::cout << std::fixed << std::setprecision(5);
    std::cout << min_val;
    std::cout << " 0" << std::endl;

    for (int i = position * N; i < position * N + N; i++) {
        std::cout << dev_solutions[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}