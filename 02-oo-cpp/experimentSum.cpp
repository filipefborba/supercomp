#include "experimentSum.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

void ExperimentSum::experiment_code() {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += this->arr[i];
    }
    this->sum = result;
}
