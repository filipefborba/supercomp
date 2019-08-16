#include "experimentPow3Mult.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

void ExperimentPow3Mult::experiment_code() {
    for (int i = 0; i < this->n; i++) {
        this->arr[i] = this->arr[i] * this->arr[i] * this->arr[i];
    }
}
