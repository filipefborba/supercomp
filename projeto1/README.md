# Projeto 1 - SIMD

## Mais informações

Para mais informações sobre o projeto, verificar o [PDF](https://github.com/filipefborba/supercomp/blob/master/projeto1/Filipe%20Borba%20-%20Projeto%201.pdf) e/ou o [Jupyter Notebook](https://github.com/filipefborba/supercomp/blob/master/projeto1/Filipe%20Borba%20-%20Projeto%201.ipynb).

## Como executar

Primeirameante, para compilar todos os executáveis com o CMake, basta usar os seguintes comandos na pasta raíz do projeto:  
```
mkdir build
cd build
cmake ..
make 
```
O comando make é responsável por compilar os executáveis. Ao atualizar o código, deve-se utilizar o comando novamente. Após isso, para iniciar cada executável, basta utilizar o comando ```./build/[nome_do_arquivo] < [nomedaentrada].txt```.

Ainda, pode-se criar testes de desempenho utilizando o ```test_generator.py```, modificando-o para cumprir os requisitos de entrada do projeto.
Os testes podem ser rodados facilmente no Jupyter Notebook. Caso queira criar um executável novo, certifique-se de que o nome do arquivo começa com "simulator". Caso queira criar uma nova entrada, utilize "entrada" para testes básicos e "supercomp" para testes de desempenho.
