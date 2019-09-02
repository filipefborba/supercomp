cmake_minimum_required(VERSION 3.0.0)
project(simulator_project VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(simulator 
    rectangle.cpp
    simulator.cpp
    main.cpp
)

add_executable(simulator_O0 
    rectangle.cpp
    simulator.cpp
    main.cpp
)
target_compile_options(simulator_O0 PUBLIC -O0)

add_executable(simulator_O1
    rectangle.cpp
    simulator.cpp
    main.cpp
)
target_compile_options(simulator_O0 PUBLIC -O1)

add_executable(simulator_O2
    rectangle.cpp
    simulator.cpp
    main.cpp
)
target_compile_options(simulator_O0 PUBLIC -O2)

add_executable(simulator_O3 
    rectangle.cpp
    simulator.cpp
    main.cpp
)
target_compile_options(simulator_O0 PUBLIC -O3)

add_executable(simulator_O3_SIMD 
    rectangle.cpp
    simulator.cpp
    main.cpp
)
target_compile_options(simulator_O3_SIMD PUBLIC -O3 -mavx -ffast-math -ftree-vectorize)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
