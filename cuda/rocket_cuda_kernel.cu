#include "rocket_trajectory_cuda.h"
#include <cuda_runtime.h>
#include "rocket_physics_cuda.cu"

/* Check whether CUDA GPU is available */
extern "C" int cuda_runtime_available_c(void)
{
    int device_count = 0;

    // Detect CUDA devices
    cudaError_t err =
        cudaGetDeviceCount(&device_count);

    // No GPU or CUDA runtime failure
    if (err != cudaSuccess || device_count <= 0)
    {
        cudaGetLastError();
        return 0;
    }

    return 1;
}

/* CUDA kernel:
   One GPU thread = one trajectory */
__global__ void simulate_population(ControlProfile *population,
                                    TrajectoryResult *results,
                                    int size)
{
    // Global thread ID
    int id =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary protection
    if (id >= size)
        return;

    // Simulate trajectory
    TrajectoryResult r =
        simulate_trajectory(&population[id], id);

    // Calculate optimization fitness
    calculate_fitness(&r);

    // Store result
    results[id] = r;
}

/* Main CUDA optimization routine */
TrajectoryResult find_best_trajectory_cuda(ControlProfile *population,
                                           int size)
{
    // GPU memory pointers
    ControlProfile *d_population;
    TrajectoryResult *d_results;

    // Allocate GPU memory
    cudaMalloc(&d_population,
               size * sizeof(ControlProfile));

    cudaMalloc(&d_results,
               size * sizeof(TrajectoryResult));

    // Copy population CPU -> GPU
    cudaMemcpy(d_population,
               population,
               size * sizeof(ControlProfile),
               cudaMemcpyHostToDevice);

    // CUDA execution configuration
    int threads = 256;
    int blocks =
        (size + threads - 1) / threads;

    // Launch GPU kernel
    simulate_population<<<blocks, threads>>>(
        d_population,
        d_results,
        size);

    // Wait for GPU execution
    cudaDeviceSynchronize();

    // Allocate CPU memory for results
    TrajectoryResult *results =
        (TrajectoryResult *)malloc(
            size * sizeof(TrajectoryResult));

    // Copy results GPU -> CPU
    cudaMemcpy(results,
               d_results,
               size * sizeof(TrajectoryResult),
               cudaMemcpyDeviceToHost);

    // Find best trajectory
    TrajectoryResult best;
    best.fitness = 1e9;

    for (int i = 0; i < size; i++)
    {
        if (results[i].fitness < best.fitness)
            best = results[i];
    }

    // Free GPU memory
    cudaFree(d_population);
    cudaFree(d_results);

    // Free CPU memory
    free(results);

    return best;
}

/* C wrapper function
   Allows CUDA code to be called from C/MPI/OpenMP code */
extern "C" TrajectoryResult find_best_trajectory_cuda_c(
    ControlProfile *population,
    int size)
{
    return find_best_trajectory_cuda(
        population,
        size);
}