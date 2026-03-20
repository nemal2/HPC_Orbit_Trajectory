#include "rocket_trajectory_cuda.h"
#include <cuda_runtime.h>
#include "rocket_physics_cuda.cu"

__global__ void simulate_population(ControlProfile *population,
                                    TrajectoryResult *results,
                                    int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size)
        return;

    TrajectoryResult r =
        simulate_trajectory(&population[id], id);

    calculate_fitness(&r);

    results[id] = r;
}

TrajectoryResult find_best_trajectory_cuda(ControlProfile *population, int size)
{
    ControlProfile *d_population;
    TrajectoryResult *d_results;

    cudaMalloc(&d_population, size * sizeof(ControlProfile));
    cudaMalloc(&d_results, size * sizeof(TrajectoryResult));

    cudaMemcpy(d_population, population,
               size * sizeof(ControlProfile),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    simulate_population<<<blocks, threads>>>(d_population, d_results, size);

    cudaDeviceSynchronize();

    TrajectoryResult *results =
        (TrajectoryResult *)malloc(size * sizeof(TrajectoryResult));

    cudaMemcpy(results, d_results,
               size * sizeof(TrajectoryResult),
               cudaMemcpyDeviceToHost);

    TrajectoryResult best;
    best.fitness = 1e9;

    for (int i = 0; i < size; i++)
        if (results[i].fitness < best.fitness)
            best = results[i];

    cudaFree(d_population);
    cudaFree(d_results);
    free(results);

    return best;
}

int main()
{
    printf("\n====================================\n");
    printf("ROCKET TRAJECTORY OPTIMIZATION CUDA\n");
    printf("====================================\n\n");

    srand(FIXED_SEED);

    ControlProfile *population =
        (ControlProfile *)malloc(
            POPULATION_SIZE * sizeof(ControlProfile));

    initialize_population(population, POPULATION_SIZE);

    double start = get_wall_time();

    TrajectoryResult best =
        find_best_trajectory_cuda(population, POPULATION_SIZE);

    double end = get_wall_time();

    print_result(&best, "BEST TRAJECTORY (CUDA)");

    printf("Execution time: %.3f sec\n", end - start);

    free(population);
}