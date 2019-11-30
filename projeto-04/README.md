# Projeto 4 - MPI

## Mais informações

Para mais informações sobre o projeto, verificar o [PDF](https://github.com/filipefborba/supercomp/blob/master/projeto-04/Filipe%20Borba%20-%20Projeto%204.pdf) e/ou o [Jupyter Notebook](https://github.com/filipefborba/supercomp/blob/master/projeto-04/Filipe%20Borba%20-%20Projeto%204.ipynb).

## Como executar

Primeirameante, para compilar todos os executáveis com o CMake, basta usar os seguintes comandos na pasta raíz do projeto:  
```
mkdir build
cd build
cmake ..
make 
```
O comando make é responsável por compilar os executáveis. Ao atualizar o código, deve-se utilizar o comando novamente. Após isso, para iniciar cada executável, basta utilizar o comando ```mpiexec -n 3 ./nome_do_arquivo < ../tests/nome_da_entrada```.


Os testes podem ser rodados facilmente utilizando o seguinte comando na pasta raíz do projeto, após a compilação:

```
./run_tests_local.sh  # para rodar o MPI localmente e salvar os resultados na pasta outputs.
./run_tests_remote.sh  # para rodar o MPI com o arquivo de hosts e salvar os resultados na pasta outputs.
```

Obs: dependendo do ambiente, será necessário tratar os outputs antes de utilizar o Jupyter Notebook.
