#!/bin/bash

# run 2opt-sol with all input files and save output
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-2opt-sol < ./tests/in10.txt > ./output/time-2opt-sol-in10
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-2opt-sol < ./tests/burma14.txt > ./output/time-2opt-sol-burma14
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-2opt-sol < ./tests/ulysses16.txt > ./output/time-2opt-sol-ulysses16
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-2opt-sol < ./tests/ulysses22.txt > ./output/time-2opt-sol-ulysses22

# run mpi-sol with all input files and save output
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-mpi-sol < ./tests/in10.txt > ./output/time-mpi-sol-in10
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-mpi-sol < ./tests/burma14.txt > ./output/time-mpi-sol-burma14
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-mpi-sol < ./tests/ulysses16.txt > ./output/time-mpi-sol-ulysses16
mpiexec -n 3 -np 2 --hostfile ./hosts ./build/time-mpi-sol < ./tests/ulysses22.txt > ./output/time-mpi-sol-ulysses22

# run tsp-seq with in10
./build/time-tsp-seq < ./tests/in10.txt > ./output/time-tsp-seq-in10
