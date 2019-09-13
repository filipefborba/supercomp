cmake_minimum_required(VERSION 3.0.0)
project(thread_safety VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(pi_mc
    pi_mc.c
    random.c
)
target_compile_options(pi_mc PUBLIC -fopenmp)
target_link_libraries(pi_mc OpenMP::OpenMP_CXX)

add_executable(pi_mc_random
    pi_mc_random.cpp
)
target_compile_options(pi_mc_random PUBLIC -fopenmp)
target_link_libraries(pi_mc_random OpenMP::OpenMP_CXX)

add_executable(pi_mc_par1
    pi_mc_par1.cpp
)
target_compile_options(pi_mc_par1 PUBLIC -fopenmp)
target_link_libraries(pi_mc_par1 OpenMP::OpenMP_CXX)

add_executable(pi_mc_par2
    pi_mc_par2.cpp
)
target_compile_options(pi_mc_par2 PUBLIC -fopenmp)
target_link_libraries(pi_mc_par2 OpenMP::OpenMP_CXX)

add_executable(pi_mc_par3
    pi_mc_par3.cpp
)
target_compile_options(pi_mc_par3 PUBLIC -fopenmp)
target_link_libraries(pi_mc_par3 OpenMP::OpenMP_CXX)