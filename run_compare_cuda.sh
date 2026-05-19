#!/bin/sh

NP="${1:-4}"

make compare_cuda && mpirun -np "$NP" ./compare_cuda
