cmake_minimum_required(VERSION 3.0.0)
project(project4_tsp VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Boost REQUIRED mpi serialization)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(2opt-sol 2opt-sol.cpp)
target_compile_options(2opt-sol PUBLIC -O3)
target_link_libraries(2opt-sol MPI::MPI_CXX Boost::mpi Boost::serialization)

add_executable(time-2opt-sol 2opt-sol.cpp)
target_compile_options(time-2opt-sol PUBLIC -O3 -DTIME)
target_link_libraries(time-2opt-sol MPI::MPI_CXX Boost::mpi Boost::serialization)

add_executable(mpi-sol mpi-sol.cpp)
target_compile_options(mpi-sol PUBLIC -O3)
target_link_libraries(mpi-sol MPI::MPI_CXX Boost::mpi Boost::serialization)

add_executable(time-mpi-sol mpi-sol.cpp)
target_compile_options(time-mpi-sol PUBLIC -O3 -DTIME)
target_link_libraries(time-mpi-sol MPI::MPI_CXX Boost::mpi Boost::serialization)

add_executable(tsp-seq tsp-seq.cpp)
target_compile_options(tsp-seq PUBLIC -O3)

add_executable(time-tsp-seq tsp-seq.cpp)
target_compile_options(time-tsp-seq PUBLIC -O3 -DTIME)
