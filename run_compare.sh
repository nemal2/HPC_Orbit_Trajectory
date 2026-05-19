#!/bin/sh

NP="${1:-4}"

if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA compiler found. Building CUDA-enabled comparison..."
    if make compare_cuda; then
        mpirun -np "$NP" ./compare_cuda
        exit $?
    fi

    echo "CUDA build failed. Falling back to non-CUDA comparison..."
fi

make compare && mpirun -np "$NP" ./compare
