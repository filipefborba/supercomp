#include <iostream>

int main() {
    int n;
    double a;
    double sum;
    std::cout << "Quantos números deseja somar para tirar a média?" << std::endl;
    std::cin >> n;
    std::cout << "Insira os números agora:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cin >> a;
        sum += a;
    }
    std::cout << "Resultado: " << sum/n << std::endl;
}