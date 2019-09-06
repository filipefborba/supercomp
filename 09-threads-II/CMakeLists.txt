cmake_minimum_required(VERSION 3.0.0)
project(pi_project VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(exemplo1 
    exemplo1.cpp
)
target_compile_options(exemplo1 PUBLIC -fopenmp)
target_link_libraries(exemplo1 OpenMP::OpenMP_CXX)

add_executable(calcula_pi 
    calcula_pi.cpp
)
target_compile_options(calcula_pi PUBLIC -fopenmp)
target_link_libraries(calcula_pi OpenMP::OpenMP_CXX)

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
