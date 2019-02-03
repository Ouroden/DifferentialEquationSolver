#!/bin/bash

#for gridsize in 10 20 30 40 50 80 100; do
for gridsize in 200 400 1000; do
	printf "#####################################################\n"
	printf "Grid size: $gridsize\n"
	printf "#####################################################\n\n"

	for number_of_cores in {1..4}; do
		printf "\nmpiexec -np $number_of_cores python solver.py $gridsize\n"
		mpiexec -np $number_of_cores python solver.py $gridsize
	done
	printf "\n"
done
