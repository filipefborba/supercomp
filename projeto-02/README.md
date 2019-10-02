# Projeto 2 - Multi-core

## Mais informações

Para mais informações sobre o projeto, verificar o [PDF](https://github.com/filipefborba/supercomp/blob/master/projeto-02/Filipe%20Borba%20-%20Projeto%202.pdf) e/ou o [Jupyter Notebook](https://github.com/filipefborba/supercomp/blob/master/projeto-02/Filipe%20Borba%20-%20Projeto%202.ipynb).

## Como executar

Primeirameante, para compilar todos os executáveis com o CMake, basta usar os seguintes comandos na pasta raíz do projeto:  
```
mkdir build
cd build
cmake ..
make 
```
O comando make é responsável por compilar os executáveis. Ao atualizar o código, deve-se utilizar o comando novamente. Após isso, para iniciar cada executável, basta utilizar o comando ```./[nome_do_executavel] < ../tests/[nome_da_entrada].txt```.

Ainda, pode-se criar testes utilizando o ```gerador.py```, como por exemplo:

```echo N | python3 gerador.py > ./tests/nome_da_entrada```,
onde N é o número de pontos desejados.


Os testes podem ser rodados facilmente no Jupyter Notebook ou utilizando o seguinte comando na pasta raíz do projeto, após a compilação:

```./build/nome_do_executavel < ../tests/nome_da_entrada```.
