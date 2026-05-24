#!/bin/sh

NP="${1:-4}"

make compare && mpirun -np "$NP" ./compare
