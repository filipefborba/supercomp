#include "experimentSum.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

void ExperimentSum::experiment_code() {
    double sum;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
}
