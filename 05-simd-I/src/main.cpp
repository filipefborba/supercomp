#include <iostream>
#include <random>
#include <vector>
#include <chrono>

std::vector<double> generate_vector(int n) {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(-10.0,10.0);
    std::vector<double> arr;
    for (int i = 0; i < n; i++) {
        arr.push_back(dist(gen));
    }
    return arr;
}

double sum_positives1(double *a, int n) {
    double sum;
    for (int i = 0; i < n; i++) {
        if (a[i] > 0) {
            sum += a[i];
        }
    }
    return sum;
}

double sum_positives2(double *a, int n) {
    double sum;
    for (int i = 0; i < n; i++) {
        (a[i] > 0) ? sum += a[i] : 0;
    }
    return sum;
}

int main () {
    int n = 10000000;
    std::vector<double> gen_vector = generate_vector(n);
    double* array = &gen_vector[0];

    auto start = std::chrono::high_resolution_clock::now();
    double result = sum_positives2(array, n);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    double duration = elapsed.count();
    std::cout << "The result was: " << result << " in " << duration << " seconds." << std::endl;
    return 0;
}