#include <iostream>
#include <cmath>

int main() {
    int n;
    std::cout << "Qual o tamanho dos vetores?" << std::endl;
    std::cin >> n;

    double *v1 = new double[n];
    std::cout << "Insira os números para o primeiro vetor agora:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cin >> v1[i];        
    }

    double *v2 = new double[n];
    std::cout << "Insira os números para o segundo vetor agora:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cin >> v2[i];    
    }

    double result;
    for (int i = 0; i < n; i++) {
        result += pow(v1[i] - v2[i], 2.0);
    }
    result = sqrt(result);

    std::cout << "Resultado: " << result << std::endl;
}