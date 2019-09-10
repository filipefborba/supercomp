cmake_minimum_required(VERSION 3.0.0)
project(pi_project_v2 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(mandel 
    mandel.cpp
)

add_executable(mandel_openmp 
    mandel.cpp
)
target_compile_options(mandel_openmp PUBLIC -fopenmp)
target_link_libraries(mandel_openmp OpenMP::OpenMP_CXX)

add_executable(calcula_pi 
    calcula_pi.cpp
)
target_compile_options(calcula_pi PUBLIC -fopenmp)
target_link_libraries(calcula_pi OpenMP::OpenMP_CXX)

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
