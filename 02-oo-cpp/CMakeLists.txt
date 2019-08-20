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
    main.cpp
)
target_compile_options(vector_operations PUBLIC -O0)

add_executable(vector_O3 
    experiment.cpp 
    experimentLog.cpp 
    experimentPow.cpp 
    experimentPow3.cpp 
    experimentPow3Mult.cpp 
    experimentSum.cpp 
    main.cpp
)
target_compile_options(vector_O3 PUBLIC -O3)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
