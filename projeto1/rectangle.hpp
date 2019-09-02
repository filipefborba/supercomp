#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <utility>

class Rectangle {
    private:
    ;
    public:
        Rectangle(double m, double wr, double hr, double x, double y, double vx, double vy, double id);
        ~Rectangle();

        double id; // id do retangulo
        bool collided; // true se colidiu, false se nao colidiu
        double m; // massa
        double wr; // largura
        double hr; // altura
        std::pair<double, double> position; // posicao inicial
        std::pair<double, double> velocity; // velocidade inicial

        void print_rect(void); // printa as propriedades do retangulo
};

#endif