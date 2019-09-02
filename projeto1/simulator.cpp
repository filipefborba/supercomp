#include "rectangle.hpp"
#include "simulator.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

Simulator::Simulator(double w1, double h1, double mu_d1, double N1, std::vector<Rectangle> rect_vector1, double dt1, double print_freq1, double max_iter1) {
    w = w1;
    h = h1;
    mu_d = mu_d1;
    accel = mu_d1*9.8;
    N = N1;
    rect_vector = rect_vector1;
    rect_vector_next = rect_vector1;
    dt = dt1;
    print_freq = print_freq1;
    max_iter = max_iter1;
    iter = 0;
    is_stable = false;
}
Simulator::~Simulator() {}

void Simulator::check_stability() {
    // Verifica se a simulacao acabou (retangulos parados)
    double v;
    for (Rectangle &rect : rect_vector) {
        v = sqrt(pow(rect.velocity.first, 2) + pow(rect.velocity.second, 2));

        // Se a velocidade de qualquer um dos corpos for maior que 0.0001, ainda está em movimento.
        if (v > 0.0001) {
            return;
        }
    }
    // Caso nao passe do if, entao todos os corpos tem v <= 0.0001
    is_stable = true;
}

void Simulator::move() {
    // Move o retangulo e modifica sua velocidade por causa do atrito
    for (Rectangle &rect : rect_vector_next) {
        rect.position.first = rect.position.first + (rect.velocity.first * dt);
        rect.position.second = rect.position.second + (rect.velocity.second * dt);

        // Verificamos se nao é 0 para não dar NAN
        if (rect.velocity.first != 0.0 || rect.velocity.second != 0.0) {
            double v = sqrt(pow(rect.velocity.first, 2) + pow(rect.velocity.second, 2));
            double theta = atan(rect.velocity.second/rect.velocity.first);
            v = v - (accel * dt);
            if (v <= 0.0) {
                v = 0.0;
            }
            rect.velocity.first = v * cos(theta);
            rect.velocity.second = v * sin(theta);
        }
    }
}

void Simulator::wall_collision() {
    // Verificacao de colisao com as paredes

    for (Rectangle &rect : rect_vector_next) {
        // Se bateu na parede direita
        if (rect.position.first + rect.wr >= w) {
            rect.collided = true;
            rect.velocity.first = -rect.velocity.first;

        // Se bateu na parede esquerda
        } else if (rect.position.first < 0) {
            rect.collided = true;
            rect.velocity.first = -rect.velocity.first;
        }

        // Se bateu na parede de cima
        if (rect.position.second >= h) {
            rect.collided = true;
            rect.velocity.second = -rect.velocity.second;

        // Se bateu na parede de baixo
        } else if (rect.position.second - rect.hr < 0) {
            rect.collided = true;
            rect.velocity.second = -rect.velocity.second;
        }
    }
}

void Simulator::rect_collision() {
    // Verificacao de colisao entre retangulos
    for (int i = 0; i < N; i++)  {
        for (int j = i + 1; j < N; j++) {
            
            // Referencia dos retangulos
            Rectangle &A = rect_vector[i];
            Rectangle &B = rect_vector[j];
            Rectangle &A_next = rect_vector_next[i];
            Rectangle &B_next = rect_vector_next[j];

            // Verifica se os retangulos estao sobrepostos
            // https://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
            if (A.id != B.id &&
            A.position.first         < B.position.first + B.wr  &&
            A.position.first + A.wr  > B.position.first         &&
            A.position.second        > B.position.second - B.hr &&
            A.position.second - A.hr < B.position.second) {
                A_next.velocity.first = ((A.velocity.first * (A.m - B.m)) + (2 * B.velocity.first * B.m) / (A.m + B.m));
                B_next.velocity.first = ((B.velocity.first * (B.m - A.m)) + (2 * A.velocity.first * A.m) / (B.m + A.m));
                A_next.velocity.second = ((A.velocity.second * (A.m - B.m)) + (2 * B.velocity.second * B.m) / (A.m + B.m));
                B_next.velocity.second = ((B.velocity.second * (B.m - A.m)) + (2 * A.velocity.second * A.m) / (B.m + A.m));

                A_next.collided = true;
                B_next.collided = true;
            }
        }
    }
}

void Simulator::check_collisions() {
    // Verificaca os retangulos que colidiram e prepara para a proxima iteracao
    for (int i = 0; i < N; i++)  {
        if (rect_vector_next[i].collided == true) {
            rect_vector_next[i].position.first = rect_vector[i].position.first;
            rect_vector_next[i].position.second = rect_vector[i].position.second;
            rect_vector_next[i].collided = false;
        }
    }
    rect_vector = rect_vector_next;
}


void Simulator::run() {
    // Roda a simulacao

    print_initialization();

    auto start = std::chrono::high_resolution_clock::now();
    while(iter < max_iter && !is_stable) {
        move();
        wall_collision();
        rect_collision();
        check_collisions();
        check_stability();

        if (int(iter) % int(print_freq) == 0) {
            print_report();
        }

        iter += 1;
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = (finish - start);
    dur = elapsed.count();

    print_final_report();
}

// ---------------------------------------------------------------------------------------------------
// Funcoes de print

void Simulator::print_initialization() {
    // Relatorio de Inicializacao
    std::cout << "----------Parameters----------" << std::endl;
    std::cout << "Simulation Grid" << std::endl;
    std::cout << "Width: " << w << " Height: " << h << " Friction: " << mu_d << std::endl;
    std::cout << "Steps (dt): " << dt << " Print Frequency: " << print_freq << " Max Iterations: " << max_iter << std::endl << std::endl;
    for (Rectangle &rect : rect_vector){
        rect.print_rect();
    }
    std::cout << "--------End Parameters--------" << std::endl << std::endl;   
}

void Simulator::print_report() {
    // Relatorio de Simulacao
    std::cout << iter << std::endl;
    for (Rectangle &rect : rect_vector){
        rect.print_rect();
    }
    std::cout << "-------------------" << std::endl;   
}

void Simulator::print_final_report() {
    std::string string_stable;
    if (is_stable) {
        string_stable = "Yes";
    } else {
        string_stable = "No";
    }
    // Relatorio Final de Simulacao (quando acabou)
    std::cout << "--------Final Report-------" << std::endl;
    std::cout << "Maximum Iterations: " << max_iter << std::endl;
    std::cout << "Is Stable? " << string_stable << std::endl     << std::endl;
    std::cout << iter << std::endl;
    for(Rectangle &rect : rect_vector){
        rect.print_rect();
    }
    std::cout << "--------End Simulation--------" << std::endl;   
    std::cout << "Total time taken in seconds: " << std::endl;
    std::cout << dur << std::endl;
    std::cout << "Number of rectangles: " << std::endl;
    std::cout << N << std::endl;
}

