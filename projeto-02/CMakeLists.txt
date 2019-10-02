cmake_minimum_required(VERSION 3.0.0)
project(project2_tsp VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(tsp_seq
    tsp_seq.cpp
)
target_compile_options(tsp_seq PUBLIC -O3)

add_executable(time_tsp_seq
    tsp_seq.cpp
)
target_compile_options(time_tsp_seq PUBLIC -O3 -DTIME)

add_executable(tsp_par
    tsp_par.cpp
)
target_compile_options(tsp_par PUBLIC -O3 -fopenmp)
target_link_libraries(tsp_par OpenMP::OpenMP_CXX)

add_executable(time_tsp_par
    tsp_par.cpp
)
target_compile_options(time_tsp_par PUBLIC -O3 -fopenmp -DTIME)
target_link_libraries(time_tsp_par OpenMP::OpenMP_CXX)

add_executable(tsp_bb
    tsp_bb.cpp
)
target_compile_options(tsp_bb PUBLIC -O3 -fopenmp)
target_link_libraries(tsp_bb OpenMP::OpenMP_CXX)

add_executable(time_tsp_bb
    tsp_bb.cpp
)
target_compile_options(time_tsp_bb PUBLIC -O3 -fopenmp -DTIME)
target_link_libraries(time_tsp_bb OpenMP::OpenMP_CXX)