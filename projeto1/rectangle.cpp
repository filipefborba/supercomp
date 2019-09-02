#include "rectangle.hpp"
#include <iostream>
#include <utility>

Rectangle::Rectangle(double mass, double width, double height, double x, double y, double vx, double vy, double n) {
    m = mass;
    wr = width;
    hr = height;
    position = std::pair<double, double>(x, y);
    velocity = std::pair<double, double>(vx, vy);
    id = n;
    collided = false;
}
Rectangle::~Rectangle() {}

void Rectangle::print_rect() {
    // std::cout << "Rectangle " << id << std::endl;
    // std::cout << "Mass: " << m << " Height: " << hr << " Width: " << wr << std::endl;
    // std::cout << "Position: (" << position.first << ", " << position.second << ")" << std::endl;
    // std::cout << "Velocity: (" << velocity.first << ", " << velocity.second << ")" << std::endl << std::endl;
    std::cout << position.first << " " << position.second << " " << velocity.first << " " <<  velocity.second << std::endl;
}


