cmake_minimum_required(VERSION 3.0.0)
project(project3_tsp LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

add_executable(random_sol
    random_sol.cu
)
target_compile_options(random_sol PUBLIC -std=c++11 -O3 )

add_executable(time_random_sol
    random_sol.cu
)
target_compile_options(time_random_sol PUBLIC -std=c++11 -O3 -DTIME)


add_executable(2opt_sol
    2opt_sol.cu
)
target_compile_options(2opt_sol PUBLIC -std=c++11 -O3)

add_executable(time_2opt_sol
    2opt_sol.cu
)
target_compile_options(time_2opt_sol PUBLIC -std=c++11 -O3 -DTIME)