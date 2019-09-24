cmake_minimum_required(VERSION 3.0.0)
project(tasks VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(race_cars
    race_cars.cpp
)
target_compile_options(race_cars PUBLIC -fopenmp)
target_link_libraries(race_cars OpenMP::OpenMP_CXX)

add_executable(fib_par
    fib_par.cpp
)
target_compile_options(fib_par PUBLIC -fopenmp)
target_link_libraries(fib_par OpenMP::OpenMP_CXX)

add_executable(pi_recursivo
    pi_recursivo.cpp
)
target_compile_options(pi_recursivo PUBLIC -fopenmp)
target_link_libraries(pi_recursivo OpenMP::OpenMP_CXX)