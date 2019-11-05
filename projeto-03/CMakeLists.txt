cmake_minimum_required(VERSION 3.0.0)
project(project3_tsp LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(random-sol
    random_sol.cu
)
target_compile_options(random-sol PUBLIC -std=c++11 -O3 )

add_executable(time-random-sol
    random_sol.cu
)
target_compile_options(time-random-sol PUBLIC -std=c++11 -O3 -DTIME)


add_executable(2opt-sol
    2opt_sol.cu
)
target_compile_options(2opt-sol PUBLIC -std=c++11 -O3)

add_executable(time-2opt-sol
    2opt_sol.cu
)
target_compile_options(time-2opt-sol PUBLIC -std=c++11 -O3 -DTIME)

add_executable(tsp-bb
    tsp_bb.cpp
)
target_compile_options(tsp-bb PUBLIC -O3 -fopenmp)
target_link_libraries(tsp-bb OpenMP::OpenMP_CXX)

add_executable(time-tsp-bb
    tsp_bb.cpp
)
target_compile_options(time-tsp-bb PUBLIC -O3 -fopenmp -DTIME)
target_link_libraries(time-tsp-bb OpenMP::OpenMP_CXX)