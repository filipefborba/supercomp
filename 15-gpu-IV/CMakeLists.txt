cmake_minimum_required(VERSION 3.0.0)
project(gpu_shared LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

find_package(OpenMP REQUIRED)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(naive128
    naive.cu
)
target_compile_options(naive128 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=128)

add_executable(naive512
    naive.cu
)
target_compile_options(naive512 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=512)

add_executable(naive1024
    naive.cu
)
target_compile_options(naive1024 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=1024)

add_executable(naive2048
    naive.cu
)
target_compile_options(naive2048 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=2048)

add_executable(tiling128
    tiling.cu
)
target_compile_options(tiling128 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=128)

add_executable(tiling512
    tiling.cu
)
target_compile_options(tiling512 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=512)

add_executable(tiling1024
    tiling.cu
)
target_compile_options(tiling1024 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=1024)

add_executable(tiling2048
    tiling.cu
)
target_compile_options(tiling2048 PUBLIC -std=c++11 -O3 -DVECTOR_SIZE=2048)

add_executable(par128
    par.cpp
)
target_compile_options(par128 PUBLIC -O3 -fopenmp -DVECTOR_SIZE=128)
target_link_libraries(par128 OpenMP::OpenMP_CXX)

add_executable(par512
    par.cpp
)
target_compile_options(par512 PUBLIC -O3 -fopenmp -DVECTOR_SIZE=512)
target_link_libraries(par512 OpenMP::OpenMP_CXX)

add_executable(par1024
    par.cpp
)
target_compile_options(par1024 PUBLIC -O3 -fopenmp -DVECTOR_SIZE=1024)
target_link_libraries(par1024 OpenMP::OpenMP_CXX)

add_executable(par2048
    par.cpp
)
target_compile_options(par2048 PUBLIC -O3 -fopenmp -DVECTOR_SIZE=2048)
target_link_libraries(par2048 OpenMP::OpenMP_CXX)

add_executable(seq128
    seq.cpp
)
target_compile_options(seq128 PUBLIC -O3 -DVECTOR_SIZE=128)

add_executable(seq512
    seq.cpp
)
target_compile_options(seq512 PUBLIC -O3 -DVECTOR_SIZE=512)

add_executable(seq1024
    seq.cpp
)
target_compile_options(seq1024 PUBLIC -O3 -DVECTOR_SIZE=1024)

add_executable(seq2048
    seq.cpp
)
target_compile_options(seq2048 PUBLIC -O3 -DVECTOR_SIZE=2048)