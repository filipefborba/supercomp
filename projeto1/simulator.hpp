#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <vector>

class Simulator {
    private:
    ;
    public:
        Simulator(double w, double h, double mu_d, double N, std::vector<Rectangle> rect_vector, double dt, double print_freq, double max_iter);
        ~Simulator();

        // Atributos
        double w; // largura do campo de simulacao
        double h; // altura do campo de simulacao
        double mu_d; // coeficiente de atrito dinamico
        double accel; // aceleracao do atrito
        double N; // numero de retangulos na simulacao
        std::vector<Rectangle> rect_vector; // vetor usado para guardar os retangulos
        std::vector<Rectangle> rect_vector_next; // vetor usado para guardar os retangulos da proxima iteracao
        double dt; // passo de simulacao
        double print_freq; // a cada 'print_freq' iteracoes, o resultado da simulacao é mostrado
        double max_iter; // numero maximo de iteracoes
        double iter; // iterador da simulacao
        bool is_stable; // verifica se a simulacao esta estavel
        double dur; // duracao em segundos da simulacao

        // Funcoes de simulacao
        void move(void);
        void wall_collision(void);
        void rect_collision(void);
        void check_collisions(void);
        void check_stability(void);
        void run(void);

        // Funcoes de print
        void print_initialization(void); // printa as propriedades iniciais da simulacao
        void print_report(void); // printa o estado atual da simulacao
        void print_final_report(void); // printa o estado final da simulacao
};

#endif