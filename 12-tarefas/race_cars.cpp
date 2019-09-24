#include <omp.h>
#include <iostream>
#include <iomanip>



int main () {
    std::cout << "I think";
    
    #pragma omp parallel 
    {
        #pragma omp master
        {
            #pragma omp task
                std::cout << " cars";
            #pragma omp task
                std::cout << " race";
        }
    }

    std::cout << " are fun" << std::endl;
    return 0;
}