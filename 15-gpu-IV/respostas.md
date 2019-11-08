## Aula 15 - GPU Shared
Filipe Borba, SuperComp

# Parte 0 - Calculando desempenho de memória

Suponha que todas threads acessem a memória global para seus elementos da matriz de entrada e assuma uma GPU com as seguintes configurações

- largura de banda DRAM de 200 GB/s
- 1.500 GFLOPS (Giga Floating Operations Per Second)

Supondo que iremos acessar dados float (4 bytes) na memória global, responda:

1. Qual a largura de banda necessária para obter desempenho máximo? 
R: A largura de banda necessária é de 6000 GB/s.
2. Dada a largura de banda de 200GB/s de nossa placa, qual o desempenho máximo que pode ser obtido acessando a memória global? R: O desempenho máximo é de 50 GFLOPS.

** Largura de Banda / 4 = FLOPS **

Ao calcular os números acima você verá que otimizações de acesso a memória são essenciais para obter desempenho máximo ao programar em GPU. Seu trabalho nesta atividade será quantificar essas diferenças de desempenho utilizando exemplos de código para multiplicação de matrizes.