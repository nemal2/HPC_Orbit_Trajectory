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

    int num_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    if (num_threads < 1)
        num_threads = omp_get_max_threads();

    printf("OpenMP Configuration:\n");
    printf("  Max threads available: %d\n", omp_get_max_threads());
    printf("  Using threads:         %d\n\n", num_threads);

    srand(FIXED_SEED);

    printf("Initializing population of %d trajectories (seed=%d)...\n",
           POPULATION_SIZE, FIXED_SEED);
    ControlProfile *population = (ControlProfile *)malloc(POPULATION_SIZE * sizeof(ControlProfile));
    initialize_population(population, POPULATION_SIZE);
    printf("✓ Population initialized\n\n");

    printf("========================================\n");
    printf("SIMULATION PARAMETERS\n");
    printf("========================================\n");
    printf("Target Altitude:     %.0f km\n", TARGET_ALTITUDE / 1000.0);
    printf("Target Velocity:     %.0f m/s\n", TARGET_VELOCITY);
    printf("Population Size:     %d\n", POPULATION_SIZE);
    printf("Control Points:      %d\n", NUM_CONTROL_POINTS);
    printf("Number of Threads:   %d\n", num_threads);
    printf("Fixed Random Seed:   %d\n", FIXED_SEED);
    printf("========================================\n\n");

    printf("Starting OPENMP optimization...\n");

    double start_time = omp_get_wtime();
    TrajectoryResult best = find_best_trajectory_openmp(population, POPULATION_SIZE, num_threads);
    double exec_time = omp_get_wtime() - start_time;

    // FIX: print_result called once, outside parallel region → no garbled output
    print_result(&best, "BEST TRAJECTORY FOUND (OPENMP)");

    printf("\n========================================\n");
    printf("PERFORMANCE: OpenMP (%d threads)\n", num_threads);
    printf("========================================\n");
    printf("Execution Time:      %.3f seconds\n", exec_time);
    printf("Trajectories/second: %.1f\n", POPULATION_SIZE / exec_time);
    printf("========================================\n\n");

    TrajectoryResult results[1] = {best};
    save_results_to_file(results, 1, "results_openmp.csv");

    printf("========================================\n");
    printf("VERIFICATION\n");
    printf("========================================\n");
    printf("Orbit Reached:  %s\n", best.reached_orbit ? "YES ✓" : "NO ✗");
    double alt_pct = fabs(best.final_altitude - TARGET_ALTITUDE) / TARGET_ALTITUDE * 100.0;
    double vel_pct = fabs(best.final_velocity - TARGET_VELOCITY) / TARGET_VELOCITY * 100.0;
    printf("Altitude Error: %.2f%%\n", alt_pct);
    printf("Velocity Error: %.2f%%\n", vel_pct);
    if (alt_pct < 10.0 && vel_pct < 10.0)
        printf("✓ Trajectory is within acceptable tolerances!\n");
    printf("========================================\n\n");

    free(population);
    printf("✓ OpenMP optimization complete!\n");
    printf("  Results saved to results_openmp.csv\n");
    printf("  Execution time: %.3f seconds with %d threads\n\n", exec_time, num_threads);
    return 0;
}