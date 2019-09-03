#include <thread>
#include <vector>
#include <iostream>
#include <chrono>

void calc_pi(double num_steps, double step, double *index) {
    double x, sum = 0.0;

    for (double i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    *index = step * sum;
}

int main() {
    // Parametros do calculo do pi
    static long num_steps = 100000000;
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
        thread_vector[i] = std::thread(calc_pi, parallel_num_steps, step, &thread_results[i]);
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
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " milisegundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;

    return 0;
}

