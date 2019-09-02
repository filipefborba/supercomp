import numpy as np

# Cria um arquivo com certo nome
file = open("supercomp_test1000.txt", "w")

# Escreve os parametros iniciais da simulacao
# largura, altura, coeficiende de atrito
file.write("100000 100000 0.2\n")

# Quantos quadrados serao criados
rng = 1000
file.write(str(rng) + "\n")

# Cria os quadrados. Atente-se ao range! modifique-os à vontade.
# massa, largura, altura, x, y, vx, vy
for i in range(rng):
    m = str(np.random.randint(1, 30))
    wr = str(np.random.randint(1, 40))
    hr = str(np.random.randint(1, 40))
    x = str(np.random.randint(0, 100000))
    y = str(np.random.randint(0, 100000))
    vx = str(np.random.randint(1, 100))
    vy = str(np.random.randint(1, 100))
    file.write(m + ' ' + wr + ' ' + hr + ' ' + x + ' ' + y + ' ' + vx + ' ' + vy + '\n')

# Por fim, escreve o resto dos parametros da simulacao e fecha o arquivo
# dt, print_freq, max_iter
file.write("0.001 10000 100000")
file.close()