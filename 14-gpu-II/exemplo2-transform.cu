#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <math.h>
#include <iostream>
#include <chrono>

struct custom_op {
    // Esse aqui teve um desempenho melhor, mas é feio demais.
    // Mais facil fazer cada operacao na mao. Fica mais legivel.
    double mean;
    
    custom_op(double m): mean(m) {}

    __host__ __device__ 
    double operator() (double el) {
        double result = el - mean;
        return result * result;
    }
};

struct ml_op {

    __host__ __device__ 
    double operator() (double el1, double el2) {
        if (el2 > (el1*1.1)) {
            return 1;
        } else {
            return 0;
        }
    }
};

struct acesso_direto {
    double *pointer_gpu;

    acesso_direto(double *pointer_gpu): pointer_gpu(pointer_gpu) {}

    __host__ __device__
    double operator() (int i) {
        if (pointer_gpu[i+1] > (pointer_gpu[i]*1.1)) {
            return 1;
        } else {
            return 0;
        }
    }
};

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


    // Aula 14
    thrust::device_vector<double> variance(size);
    start_time = std::chrono::high_resolution_clock::now();

    // Implementacao 1

    // thrust::transform(result.begin(), result.end(), thrust::make_constant_iterator(mean), variance.begin(), thrust::minus<double>());
    // thrust::transform(variance.begin(), variance.end(), variance.begin(), variance.begin(), thrust::multiplies<double>());

    // Implementacao 2 (struct)
    thrust::transform(result.begin(), result.end(), variance.begin(), custom_op(mean));

    end_time = std::chrono::high_resolution_clock::now();
    runtime = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cerr << runtime.count() << " us" << std::endl;

    double var = thrust::reduce(variance.begin(), variance.end(), (double) 0, thrust::plus<double>()) / (size - 1);
    std::cout << "Variancia " << var << std::endl;

    thrust::device_vector<double> ml_vec(size);

    // Implementacao 1

    // thrust::transform(dev_aapl.begin(), dev_aapl.end()-1, dev_aapl.begin()+1, ml_vec.begin(), ml_op());

    // Implementacao 2
    auto index = thrust::make_counting_iterator(0);
    acesso_direto op(thrust::raw_pointer_cast(dev_aapl.data()));
    thrust::transform(index, index + dev_aapl.size()-1, ml_vec.begin(), op);

    for (thrust::device_vector<double>::iterator i = ml_vec.begin(); i != ml_vec.end(); i++) {
        std::cout << *i << " ";
    }

}
