#include "experimentPow3.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

void ExperimentPow3::experiment_code() {
    for (int i = 0; i < this->n; i++) {
        this->arr[i] = pow(this->arr[i], 3.0);
    }
}
