cmake_minimum_required(VERSION 3.0.0)
project(pi_project VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(exemplo1_threads 
    exemplo1-threads.cpp
)
target_compile_options(exemplo1_threads PUBLIC -lpthread)
target_link_libraries(exemplo1_threads Threads::Threads)

add_executable(pi_numeric_integration 
    pi-numeric-integration.cpp
)
target_compile_options(pi_numeric_integration PUBLIC -pthread)

add_executable(calcula_pi 
    calcula_pi.cpp
)
target_compile_options(calcula_pi PUBLIC -pthread)
target_link_libraries(calcula_pi Threads::Threads)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
