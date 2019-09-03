#include <thread>
#include <iostream>
#include <vector>
#include <cmath>

void funcao_rodando_em_paralelo(int tid, int *index) {
    *index = pow(tid, 2);
}

int main() {
    int num_threads = std::thread::hardware_concurrency(); // Numero total de threads
    std::vector<std::thread> thread_vector(num_threads); // Vetor que guarda as threads
    std::vector<int> thread_squared_ids(num_threads); // Vetor que guarda os resultados

    std::cout << "Max number of threads: " << num_threads << std::endl;

    // Cria thread e a executa.
    // Primeiro argumento é a função a ser executada.
    // Os argumentos em seguida são passados diretamente
    // para a função passada no primeiro argumento.
    for (int i = 0; i < num_threads; i++) {
        thread_vector[i] = std::thread(funcao_rodando_em_paralelo, i, &thread_squared_ids[i]);
    }
    
    // Espera até que a função acabe de executar.
    for (int i = 0; i < num_threads; i++) {
        thread_vector[i].join();
    }

    // Recebe o id ao quadrado de cada thread e imprime
    for (int i = 0; i < num_threads; i++) {
        std::cout << "Thread: " << i << " | Result: " << thread_squared_ids[i] << std::endl;
    }
    
    // Soma de todos os elementos e resultado final
    int sum = 0;
    for (int &n : thread_squared_ids) {
        sum += n;
    }
    std::cout << "Final Result: " << sum << std::endl;

    return 0;
}
