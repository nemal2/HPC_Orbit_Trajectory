#include "rocket_trajectory.h"
#include <omp.h>

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

    free(population);
    return 0;
}