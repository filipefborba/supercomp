#include "experimentLog.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

void ExperimentLog::experiment_code() {
    for (int i = 0; i < this->n; i++) {
        this->arr[i] = log(this->arr[i]);
    }
}
