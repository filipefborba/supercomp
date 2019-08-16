#include "experiment.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

Experiment::Experiment() : n(10) {}
Experiment::Experiment(int size) : n(size) {}
Experiment::~Experiment() {}

double* Experiment::generate_vector(int n) {
    srand(42);
    double* arr = new double[n];
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }
    this->arr = arr;
    return arr;
}

double Experiment::duration() {
    return this->dur;
}

void Experiment::run() {
    auto start = std::chrono::high_resolution_clock::now();
    this->experiment_code();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    this->dur = elapsed.count()*1000;
}

void Experiment::experiment_code() {
    std::cout << "Class Experiment; experiment_code()" << std::endl;
}

Experiment::operator double() {
    return this->duration();
}

bool Experiment::operator< (Experiment &e) {
    if (this->dur < e.dur && this->n == e.n) {
        return true;
    } else {
        return false;
    }
}

bool Experiment::operator < (double duration) {
    if (this->dur < duration) {
        return true;
    } else {
        return false;
    }
}
