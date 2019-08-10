#include <iostream>
#include <cmath>
#include <chrono>

int main() {
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    int n;
    std::cin >> n;
    double *v1 = new double[n];
    double *v2 = new double[n];
    for (int i = 0; i < n; i++) {
        std::cin >> v1[i];
    }
    for (int i = 0; i < n; i++) {
        std::cin >> v2[i];
    }

    double result;
    for (int i = 0; i < n; i++) {
        result += pow(v1[i] - v2[i], 2.0);
    }
    result = sqrt(result);

    std::cout << "Resultado: " << result << std::endl;
    // Record end time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Time: " << elapsed.count()*1000 << std::endl;
}