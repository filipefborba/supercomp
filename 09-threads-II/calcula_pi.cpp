#include <omp.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

double pi_seq (long num_steps) {
    int i;
    double x, pi, sum = 0.0;
    double step = 1.0 / (double) num_steps;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end_time - start_time);
    
    std::cout << "-------------------------" << std::endl;
    std::cout << "pi_seq" << std::endl;
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
    std::cout << "-------------------------" << std::endl;

    return pi;
}

void calc_pi(long i, long num_steps, double step, double *index) {
    double x, sum = 0.0;

    for (i; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    *index = step * sum;
}

double calc_pi_double(long i, long num_steps, double step) {
    double x, sum = 0.0;

    for (i; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    return step * sum;
}

double pi_threads_raiz (long num_steps) {
    // Parametros do calculo do pi
    double step = 1.0 / (double) num_steps;

    // Parametros threads e declaracoes
    int num_threads = 4; // Numero total de threads
    std::vector<std::thread> thread_vector(num_threads); // Vetor que guarda as threads
    std::vector<double> thread_results(num_threads); // Vetor que guarda os resultados
    double parallel_num_steps = (double) num_steps / (double) num_threads;

    // Inicio da contagem do tempo
    auto start_time = std::chrono::high_resolution_clock::now();

    // Criacao dinamica das threads
    for (int i = 0; i < num_threads; i++) {
        thread_vector[i] = std::thread(calc_pi, parallel_num_steps * i, parallel_num_steps * (i+1), step, &thread_results[i]);
    }

    // Espera até que todas as threads acabem de executar.
    for (int i = 0; i < num_threads; i++) {
        thread_vector[i].join();
    }

    // Calculo final da contagem de tempo. Obtemos o tempo de execucao aqui
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end_time - start_time);

    // Soma de todos os elementos do vetor e resultado final
    double pi = 0;
    for (double &n : thread_results) {
        pi += n;
    }
    
    // Print dos resultados
    std::cout << "-------------------------" << std::endl;
    std::cout << "pi_threads_raiz" << std::endl;
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
    std::cout << "-------------------------" << std::endl;
    return pi;
}

double pi_omp_parallel_local (long num_steps) {
    int num_threads = omp_get_max_threads(); // Numero total de threads
    std::vector<double> thread_results(num_threads); // Vetor que guarda os resultados
    double parallel_num_steps = (double) num_steps / (double) num_threads; // total de passos de cada thread
    double step = 1.0 / (double) num_steps;

    double pi = 0;

    // Inicio da contagem do tempo
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // id da thread
        calc_pi(parallel_num_steps * thread_id, parallel_num_steps * (thread_id+1), step, &thread_results[thread_id]);
    }
    // Calculo final da contagem de tempo. Obtemos o tempo de execucao aqui
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end_time - start_time);

    // Soma de todos os elementos do vetor e resultado final
    for (double &n : thread_results) {
        pi += n;
    }
    
    // Print dos resultados
    std::cout << "-------------------------" << std::endl;
    std::cout << "pi_omp_parallel_local" << std::endl;
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
    std::cout << "-------------------------" << std::endl;

    return pi;

}

double pi_omp_parallel_atomic (long num_steps) {
    int num_threads = omp_get_max_threads(); // Numero total de threads
    double parallel_num_steps = (double) num_steps / (double) num_threads; // total de passos de cada thread
    double step = 1.0 / (double) num_steps;

    double pi = 0;

    // Inicio da contagem do tempo
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // id da thread

        #pragma omp atomic
        pi += calc_pi_double(parallel_num_steps * thread_id, parallel_num_steps * (thread_id+1), step);
    }
    // Calculo final da contagem de tempo. Obtemos o tempo de execucao aqui
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end_time - start_time);
    
    // Print dos resultados
    std::cout << "-------------------------" << std::endl;
    std::cout << "pi_omp_parallel_atomic" << std::endl;
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
    std::cout << "-------------------------" << std::endl;

    return pi;
}

double pi_omp_parallel_critical (long num_steps) {
    int num_threads = omp_get_max_threads(); // Numero total de threads
    double parallel_num_steps = (double) num_steps / (double) num_threads; // total de passos de cada thread
    double step = 1.0 / (double) num_steps;

    double pi = 0;

    // Inicio da contagem do tempo
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // id da thread
        double sum = calc_pi_double(parallel_num_steps * thread_id, parallel_num_steps * (thread_id+1), step);

        #pragma omp critical
        pi += sum;
    }
    // Calculo final da contagem de tempo. Obtemos o tempo de execucao aqui
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end_time - start_time);
    
    // Print dos resultados
    std::cout << "-------------------------" << std::endl;
    std::cout << "pi_omp_parallel_critical" << std::endl;
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
    std::cout << "-------------------------" << std::endl;

    return pi;
}

double pi_omp_parallel_critical_errado (long num_steps) {
    int num_threads = omp_get_max_threads(); // Numero total de threads
    double parallel_num_steps = (double) num_steps / (double) num_threads; // total de passos de cada thread
    double step = 1.0 / (double) num_steps;

    double pi = 0;

    // Inicio da contagem do tempo
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // id da thread

        #pragma omp critical
        pi += calc_pi_double(parallel_num_steps * thread_id, parallel_num_steps * (thread_id+1), step);
    }
    // Calculo final da contagem de tempo. Obtemos o tempo de execucao aqui
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds> (end_time - start_time);
    
    // Print dos resultados
    std::cout << "-------------------------" << std::endl;
    std::cout << "pi_omp_parallel_critical" << std::endl;
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
    std::cout << "-------------------------" << std::endl;

    return pi;
}

int main() {
    static long num_steps = 100000000;
    double pi = pi_seq(num_steps);
    double pi_thread = pi_threads_raiz(num_steps);
    double pi_omp = pi_omp_parallel_local(num_steps);
    double pi_omp_atomic = pi_omp_parallel_atomic(num_steps);
    double pi_omp_critical = pi_omp_parallel_critical(num_steps);
    double pi_omp_critical_errado = pi_omp_parallel_critical_errado(num_steps);
}

