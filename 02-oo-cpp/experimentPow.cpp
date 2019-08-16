#include "experimentPow.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

void ExperimentPow::experiment_code() {
    for (int i = 0; i < this->n; i++) {
        this->arr[i] = pow(this->arr[i], 2.0);
    }
}
