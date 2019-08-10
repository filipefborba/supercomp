#include <iostream>
#include <cmath>
#include <chrono>

double* gera_vetor(int n) {
    srand(42);
    double* arr = new double[n];
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }
    return arr;
}

double* log(double *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = log(arr[i]);
    }
}

double* pow(double *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = pow(arr[i], 2.0);
    }
}

double* pow3(double *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = pow(arr[i], 3.0);
    }
}

double* pow3mult(double *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = arr[i] * arr[i] * arr[i];
    }
}

double sum(double *arr, int n) {
    double sum;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    for (int n = 100; n <= 100000000; n*=10) {
        std::cout << n << std::endl;

        // Teste log
        double* v = gera_vetor(n);
        auto start = std::chrono::high_resolution_clock::now();
        log(v, n);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Log - Time: " << elapsed.count()*1000 << std::endl;

        // Teste pow
        v = gera_vetor(n);
        start = std::chrono::high_resolution_clock::now();
        pow(v, n);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        std::cout << "Pow - Time: " << elapsed.count()*1000 << std::endl;

        // Teste pow3
        v = gera_vetor(n);
        start = std::chrono::high_resolution_clock::now();
        pow3(v, n);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        std::cout << "Pow3 - Time: " << elapsed.count()*1000 << std::endl;

        // Teste pow3mult
        v = gera_vetor(n);
        start = std::chrono::high_resolution_clock::now();
        pow3mult(v, n);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        std::cout << "Pow3Mult - Time: " << elapsed.count()*1000 << std::endl;

        // Teste sum
        v = gera_vetor(n);
        start = std::chrono::high_resolution_clock::now();
        double s = sum(v, n);
        finish = std::chrono::high_resolution_clock::now();
        elapsed = finish - start;
        std::cout << "Sum - Time: " << elapsed.count()*1000 << std::endl << std::endl;
        delete[] v;
    }
}

//g++ exercicio4.cpp -o exercicio4 && ./exercicio4
