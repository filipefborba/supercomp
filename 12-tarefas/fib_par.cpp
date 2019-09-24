#include <iostream>
#include <omp.h>
#include <chrono>
#include <math.h>

int fib(int n) {
    int x, y;
    if (n < 2) return n;
    x = fib(n - 1);
    y = fib(n - 2);
    return (x + y);
}

int fib_par2(int n, int t) {
    int x, y;
    if (n < 2) return n;

    if (t == 0) {
        x = fib(n - 1);
        y = fib(n - 2);
    } else {
        #pragma omp parallel
        {

            #pragma omp master
            {
                #pragma omp task
                x = fib_par2(n - 1, t - 1);

                #pragma omp task
                y = fib_par2(n - 2, t - 1);
            }
            #pragma omp taskwait
        }
    }

    return (x + y);
}

int main() {
    int NW = 45;
    int max_threads = omp_get_max_threads();

    // Sequencial
    auto start_time = std::chrono::high_resolution_clock::now();
    int f = fib(NW);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << f << std::endl;
    std::cout << runtime.count() << " milisegundo(s)." << std::endl;

    // Paralelo
    start_time = std::chrono::high_resolution_clock::now();
    int f2 = fib_par2(NW, max_threads);
    end_time = std::chrono::high_resolution_clock::now();
    runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << f2 << std::endl;
    std::cout << runtime.count() << " milisegundo(s)." << std::endl;
}