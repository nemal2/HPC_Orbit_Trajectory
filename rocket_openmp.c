#include "rocket_trajectory.h"
#include <omp.h>

TrajectoryResult find_best_trajectory_serial_test(ControlProfile *population, int size)
{
    TrajectoryResult best;
    best.fitness = 1e9;

    for (int i = 0; i < size; i++)
    {
        TrajectoryResult result = simulate_trajectory(&population[i], i);
        calculate_fitness(&result);

        if (result.fitness < best.fitness)
            best = result;

        if ((i + 1) % 100 == 0)
            printf("Tested %d/%d trajectories\n", i + 1, size);
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
    TrajectoryResult best = find_best_trajectory_serial_test(population, POPULATION_SIZE);
    double exec_time = omp_get_wtime() - start_time;

    printf("\nBest trajectory ID: %d\n", best.trajectory_id);
    printf("Fuel consumed: %.2f kg\n", best.fuel_consumed);
    printf("Execution time: %.3f seconds\n", exec_time);

    free(population);

    return 0;
}