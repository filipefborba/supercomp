#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <omp.h>
#include <chrono>

int main() {
    thrust::host_vector<double> host;
    double input;
    double size = 0;

    while (!std::cin.eof()) {
        std::cin >> input;
        host.push_back(input);
        size++;
    }

    double inf = std::numeric_limits<double>::infinity();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    thrust::device_vector<double> dev(host);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);\
    std::cerr << runtime.count() << " us" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    double mean_historical = thrust::reduce(dev.begin(), dev.end(), (double) 0, thrust::plus<double>()) / size;
    double mean_year = thrust::reduce(dev.begin(), dev.begin()+365, (double) 0, thrust::plus<double>()) / 365;
    std::cout << "Média histórica " << mean_historical << std::endl;
    std::cout << "Média último ano " << mean_year << std::endl;
    double lowest_historical = thrust::reduce(dev.begin(), dev.end(), inf, thrust::minimum<double>());
    std::cout << "Menor histórico " << lowest_historical << std::endl;
    double highest_historical = thrust::reduce(dev.begin(), dev.end(), (double) 0, thrust::maximum<double>());
    std::cout << "Maior histórico " << highest_historical << std::endl;
    double lowest_year = thrust::reduce(dev.begin(), dev.begin()+365, inf, thrust::minimum<double>());
    std::cout << "Menor último ano " << lowest_year << std::endl;
    double highest_year = thrust::reduce(dev.begin(), dev.begin()+365, (double) 0, thrust::maximum<double>());
    std::cout << "Maior último ano " << highest_year << std::endl;
    end_time = std::chrono::high_resolution_clock::now();
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);\
    std::cerr << runtime.count() << " us" << std::endl;

    // OPENMP
    // auto start_time = std::chrono::high_resolution_clock::now();
    // thrust::device_vector<double> dev(host);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);\
    // std::cerr << runtime.count() << " us" << std::endl;

    // start_time = std::chrono::high_resolution_clock::now();
    // double mean_historical = thrust::reduce(thrust::host, host.begin(), host.end(), (double) 0, thrust::plus<double>()) / size;
    // double mean_year = thrust::reduce(thrust::host, host.begin(), host.begin()+365, (double) 0, thrust::plus<double>()) / 365;
    // std::cout << "Média histórica " << mean_historical << std::endl;
    // std::cout << "Média último ano " << mean_year << std::endl;
    // double lowest_historical = thrust::reduce(thrust::host, host.begin(), host.end(), inf, thrust::minimum<double>());
    // std::cout << "Menor histórico " << lowest_historical << std::endl;
    // double highest_historical = thrust::reduce(thrust::host, host.begin(), host.end(), (double) 0, thrust::maximum<double>());
    // std::cout << "Maior histórico " << highest_historical << std::endl;
    // double lowest_year = thrust::reduce(thrust::host, host.begin(), host.begin()+365, inf, thrust::minimum<double>());
    // std::cout << "Menor último ano " << lowest_year << std::endl;
    // double highest_year = thrust::reduce(thrust::host, host.begin(), host.begin()+365, (double) 0, thrust::maximum<double>());
    // std::cout << "Maior último ano " << highest_year << std::endl;
    // end_time = std::chrono::high_resolution_clock::now();
    // runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);\
    // std::cerr << runtime.count() << " us" << std::endl;

    return 0;
}
