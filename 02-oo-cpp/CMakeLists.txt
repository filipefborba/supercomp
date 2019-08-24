cmake_minimum_required(VERSION 3.0.0)
project(experiments_project VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(vector_operations 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    experimentSumPositives.cpp
    main.cpp
)

add_executable(vector_O0 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    experimentSumPositives.cpp 
    main.cpp
)
target_compile_options(vector_O0 PUBLIC -O0)

add_executable(vector_O1 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    experimentSumPositives.cpp 
    main.cpp
)
target_compile_options(vector_O1 PUBLIC -O1)

add_executable(vector_O2 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    experimentSumPositives.cpp 
    main.cpp
)
target_compile_options(vector_O2 PUBLIC -O2)

add_executable(vector_O3 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    experimentSumPositives.cpp 
    main.cpp
)
target_compile_options(vector_O3 PUBLIC -O3)

add_executable(vector_O3_SIMD 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    experimentSumPositives.cpp 
    main.cpp
)
target_compile_options(vector_O3_SIMD PUBLIC -O3 -mavx -ffast-math -ftree-vectorize)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
