#include "rectangle.hpp"
#include "simulator.hpp"

#include <iostream>
#include <utility>
#include <vector>

int main() {

    // Inicializacao
    double w; std::cin >> w; // largura do campo de simulacao
    double h; std::cin >> h; // altura do campo de simulacao
    double mu_d; std::cin >> mu_d; // coeficiente de atrito dinamico
    double N; std::cin >> N; // numero de retangulos na simulacao
    std::vector<Rectangle> rect_vector; // vetor usado para guardar os retangulos
    double mass, width, height, x, y, vx, vy; // parametros do retangulo
    for(int i = 0; i < N; i++) {
        std::cin >> mass; std::cin >> width; std::cin >> height;
        std::cin >> x; std::cin >> y; std::cin >> vx; std::cin >> vy;
        rect_vector.push_back(Rectangle(mass, width, height, x, y, vx, vy, i));
    }
    double dt; std::cin >> dt; // passo de simulacao
    double print_freq; std::cin >> print_freq; // a cada 'print_freq' iteracoes, o resultado da simulacao é mostrado
    double max_iter; std::cin >> max_iter; // numero maximo de iteracoes

    Simulator simulator = Simulator(w, h, mu_d, N, rect_vector, dt, print_freq, max_iter);
    simulator.run();

    return 0;
}