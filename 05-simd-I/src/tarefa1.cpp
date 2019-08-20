/*
Num registrador %xmm0 de 128 bits, podemos armazenar:
16 chars,
8 shorts,
8 ints,
4 longs,
4 floats,
2 doubles
Para registradores %ymm é só dobrar os valores acima.
*/

/* Ao rodar com SIMD, podemos perceber instruções diferentes no 
Assembly, como por exemplo o uso de 'vpxor', 'vpaddd', entre outros. */

// Original: Nicolas Brailovsky

#define SIZE (400)
long sum(int v[SIZE]) throw() {
    double *d = new double[100];
    int s = 0; // este exemplo eh didatico. soma de ints deveria ser long ;)
    for (unsigned i=0; i<SIZE; i++) s += v[i];
    return s;
}
