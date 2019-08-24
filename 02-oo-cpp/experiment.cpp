#include "experiment.hpp"
#include <iostream>
#include <random>
#include <utility>
#include <cmath>
#include <chrono>

Experiment::Experiment() : n(10) {}
Experiment::~Experiment() {}

std::vector<double> Experiment::generate_vector(int n) {
    std::vector<double> arr;
    std::default_random_engine gen(2);

    this->n = n;
    std::normal_distribution<double> d(5, sqrt(0.5));
    for (int i = 0; i < n; i++) {
        arr.push_back(d(gen));
    }
    this->arr = arr;
    return arr;
}

double Experiment::duration() {
    return this->dur;
}

std::pair<double, double> Experiment::run() {
    std::vector<double> times_arr;
    for (int i = 10; i > 0; i--) {
        auto start = std::chrono::high_resolution_clock::now();
        this->experiment_code();
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        this->dur = elapsed.count()*1000;
        times_arr.push_back(elapsed.count()*1000);
    }

    // https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boost
    double mean = std::accumulate(times_arr.begin(), times_arr.end(), 0.0)/times_arr.size();
    double sq_sum = std::inner_product(times_arr.begin(), times_arr.end(), times_arr.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times_arr.size() - mean * mean);
    return std::pair<double, double>(mean, stdev);
}

void Experiment::experiment_code() {
    std::cout << "Class Experiment; experiment_code()" << std::endl;
}

Experiment::operator double() {
    return this->duration();
}

bool Experiment::operator < (Experiment &e) {
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
