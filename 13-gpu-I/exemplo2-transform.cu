#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <iostream>
#include <chrono>

int main() {
    thrust::host_vector<double> host_aapl;
    thrust::host_vector<double> host_msft;
    double input;
    double size = 0;

    while (!std::cin.eof()) {
        std::cin >> input;
        host_aapl.push_back(input);
        std::cin >> input;
        host_msft.push_back(input);
        size++;
    }

    thrust::device_vector<double> dev_aapl(host_aapl);
    thrust::device_vector<double> dev_msft(host_msft);
    thrust::device_vector<double> result(size);

    auto start_time = std::chrono::high_resolution_clock::now();
    thrust::transform(dev_aapl.begin(), dev_aapl.end(), dev_msft.begin(), result.begin(), thrust::minus<double>());
    double mean = thrust::reduce(result.begin(), result.end(), (double) 0, thrust::plus<double>()) / size;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cerr << runtime.count() << " us" << std::endl;
    std::cout << "Diferença Média " << mean << std::endl;
}
