#include "rocket_trajectory.h"
#include <omp.h>

TrajectoryResult find_best_trajectory_openmp(ControlProfile *population,
                                             int size, int num_threads)
{
    TrajectoryResult best;
    best.fitness = 1e9;

    omp_set_num_threads(num_threads);
    printf("Using %d OpenMP threads\n", num_threads);
    printf("Each thread will test ~%d trajectories\n\n", size / num_threads);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int local_count = 0;
        TrajectoryResult local_best;
        local_best.fitness = 1e9;

// CHANGED: Add dynamic scheduling for better load balancing
#pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < size; i++)
        {
            TrajectoryResult result = simulate_trajectory(&population[i], i);
            calculate_fitness(&result);

            if (result.fitness < local_best.fitness)
                local_best = result;

            local_count++;

            // Progress reporting from thread 0
            if (thread_id == 0 && local_count % 50 == 0)
                printf("Thread 0: Processed ~%d chunks\n", local_count);
        }

#pragma omp critical
        {
            if (local_best.fitness < best.fitness)
            {
                best = local_best;
                printf("Thread %d found better trajectory: fitness=%.4f\n",
                       thread_id, local_best.fitness);
            }
        }
    }

    return best;
}

int main(int argc, char *argv[])
{
    printf("ROCKET TRAJECTORY OPTIMIZATION - OPENMP VERSION\n\n");

    // Get number of threads from command line
    int num_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();

    printf("Using %d OpenMP threads\n", num_threads);

    // Initialize with fixed seed
    srand(FIXED_SEED);

    // Create population
    ControlProfile *population = malloc(POPULATION_SIZE * sizeof(ControlProfile));
    initialize_population(population, POPULATION_SIZE);

    printf("Population of %d trajectories initialized\n", POPULATION_SIZE);

    double start_time = omp_get_wtime();
    TrajectoryResult best = find_best_trajectory_openmp(population, POPULATION_SIZE, num_threads);
    double exec_time = omp_get_wtime() - start_time;

    printf("\nBest trajectory ID: %d\n", best.trajectory_id);
    printf("Fuel consumed: %.2f kg\n", best.fuel_consumed);
    printf("Execution time: %.3f seconds\n", exec_time);

    free(population);

    return 0;
}