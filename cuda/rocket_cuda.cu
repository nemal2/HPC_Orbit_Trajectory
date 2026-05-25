#include "rocket_trajectory_cuda.h"
#include <cuda_runtime.h>
#include "rocket_physics_cuda.cu"

/* GPU kernel:
   Each CUDA thread simulates one trajectory */
__global__ void simulate_population(ControlProfile *population,
                                    TrajectoryResult *results,
                                    int size)
{
    // Unique thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Prevent out-of-bounds access
    if (id >= size)
        return;

    // Run trajectory simulation
    TrajectoryResult r =
        simulate_trajectory(&population[id], id);

    // Compute fitness
    calculate_fitness(&r);

    // Store result
    results[id] = r;
}

/* Main CUDA optimization function */
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

    // Copy trajectories CPU -> GPU
    cudaMemcpy(d_population,
               population,
               size * sizeof(ControlProfile),
               cudaMemcpyHostToDevice);

    // CUDA launch configuration
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch GPU kernel
    simulate_population<<<blocks, threads>>>(
        d_population,
        d_results,
        size);

    // Wait for GPU completion
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

int main()
{
    printf("\n====================================\n");
    printf("ROCKET TRAJECTORY OPTIMIZATION CUDA\n");
    printf("====================================\n\n");

    // Fixed seed for reproducibility
    srand(FIXED_SEED);

    // Allocate population
    ControlProfile *population =
        (ControlProfile *)malloc(
            POPULATION_SIZE * sizeof(ControlProfile));

    // Initialize trajectories
    initialize_population(population,
                          POPULATION_SIZE);

    // Start timing
    double start = get_wall_time();

    // Run CUDA optimization
    TrajectoryResult best =
        find_best_trajectory_cuda(
            population,
            POPULATION_SIZE);

    // Stop timing
    double end = get_wall_time();

    // Print best trajectory
    print_result(&best,
                 "BEST TRAJECTORY (CUDA)");

    // Print execution time
    printf("Execution time: %.3f sec\n",
           end - start);

    // Cleanup
    free(population);

    return 0;
}