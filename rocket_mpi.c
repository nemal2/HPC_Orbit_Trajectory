/*
 * MPI VERSION - Rocket Trajectory Optimization
 
 * Compile: mpicc -o rocket_mpi rocket_mpi.c rocket_physics.c rocket_utils.c -lm -O3
 * Run:     mpirun -np 4 ./rocket_mpi
 */

#include "rocket_trajectory.h"
#include <mpi.h>

TrajectoryResult find_best_trajectory_mpi(ControlProfile *population,
                                          int size, int rank, int num_procs) {
    TrajectoryResult local_best, global_best;
    local_best.fitness  = 1e9;
    global_best.fitness = 1e9;

    int per_proc   = size / num_procs;
    int start_idx  = rank * per_proc;
    int end_idx    = (rank == num_procs-1) ? size : (rank+1) * per_proc;
    int local_count= end_idx - start_idx;

    if (rank == 0) {
        printf("  MPI Configuration:\n");
        printf("    Total processes:        %d\n", num_procs);
        printf("    Trajectories per proc:  ~%d\n\n", per_proc);
    }

    // FIX: Every process generates the FULL population with the SAME seed,
    // then only works on its assigned slice.  This avoids a large MPI_Send
    // and guarantees bit-identical trajectories across all runs.
    if (rank != 0) {
        // Regenerate population on non-root processes
        srand(FIXED_SEED);
        initialize_population(population, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("  Process %d: Testing trajectories %d to %d (%d total)\n",
           rank, start_idx, end_idx-1, local_count);

    for (int i = 0; i < local_count; i++) {
        int global_id = start_idx + i;
        TrajectoryResult result = simulate_trajectory(&population[global_id], global_id);
        calculate_fitness(&result);

        if (result.fitness < local_best.fitness)
            local_best = result;

        if (rank == 0 && (i+1) % 50 == 0)
            printf("  Process 0: %d/%d (%.0f%%)\n",
                   i+1, local_count, (i+1)*100.0/local_count);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("  Process %d: Local best fitness = %.4f\n", rank, local_best.fitness);

    // Gather results at root
    if (rank == 0) {
        global_best = local_best;
        for (int i = 1; i < num_procs; i++) {
            TrajectoryResult remote;
            MPI_Recv(&remote, sizeof(TrajectoryResult), MPI_BYTE,
                     i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (remote.fitness < global_best.fitness) {
                global_best = remote;
                printf("  Process 0: Process %d has better solution (fitness=%.4f)\n",
                       i, remote.fitness);
            }
        }
    } else {
        MPI_Send(&local_best, sizeof(TrajectoryResult), MPI_BYTE,
                 0, 1, MPI_COMM_WORLD);
    }

    MPI_Bcast(&global_best, sizeof(TrajectoryResult), MPI_BYTE, 0, MPI_COMM_WORLD);
    return global_best;
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║   ROCKET TRAJECTORY OPTIMIZATION - MPI VERSION (FIXED)     ║\n");
        printf("║   Distributed Memory Parallel Computing                    ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n\n");
    }

    // FIXED: same seed on every process
    srand(FIXED_SEED);

    ControlProfile *population = (ControlProfile*)malloc(POPULATION_SIZE * sizeof(ControlProfile));
    initialize_population(population, POPULATION_SIZE);

    if (rank == 0) {
        printf("Population of %d trajectories initialized (seed=%d)\n\n",
               POPULATION_SIZE, FIXED_SEED);
        printf("========================================\n");
        printf("SIMULATION PARAMETERS\n");
        printf("========================================\n");
        printf("Target Altitude:     %.0f km\n",  TARGET_ALTITUDE / 1000.0);
        printf("Target Velocity:     %.0f m/s\n", TARGET_VELOCITY);
        printf("Population Size:     %d\n",        POPULATION_SIZE);
        printf("MPI Processes:       %d\n",        num_procs);
        printf("Fixed Random Seed:   %d\n",        FIXED_SEED);
        printf("========================================\n\n");
        printf("Starting MPI optimization...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    TrajectoryResult best = find_best_trajectory_mpi(population, POPULATION_SIZE,
                                                     rank, num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    double exec_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        print_result(&best, "BEST TRAJECTORY FOUND (MPI)");

        printf("\n========================================\n");
        printf("PERFORMANCE: MPI (%d processes)\n", num_procs);
        printf("========================================\n");
        printf("Execution Time:      %.3f seconds\n", exec_time);
        printf("Trajectories/second: %.1f\n", POPULATION_SIZE / exec_time);
        printf("Speedup potential:   %.2fx (ideal with %d processes)\n",
               (double)num_procs, num_procs);
        printf("========================================\n\n");

        TrajectoryResult results[1] = {best};
        save_results_to_file(results, 1, "results_mpi.csv");

        printf("========================================\n");
        printf("VERIFICATION\n");
        printf("========================================\n");
        printf("Orbit Reached:  %s\n", best.reached_orbit ? "YES ✓" : "NO ✗");
        double a = fabs(best.final_altitude - TARGET_ALTITUDE) / TARGET_ALTITUDE * 100.0;
        double v = fabs(best.final_velocity  - TARGET_VELOCITY)  / TARGET_VELOCITY  * 100.0;
        printf("Altitude Error: %.2f%%\n", a);
        printf("Velocity Error: %.2f%%\n", v);
        if (a < 10.0 && v < 10.0)
            printf("Trajectory is within acceptable tolerances!\n");
        printf("========================================\n\n");

        printf("  MPI optimization complete!\n");
        printf("  Results saved to results_mpi.csv\n");
        printf("  Execution time: %.3f seconds with %d processes\n\n",
               exec_time, num_procs);
    }

    free(population);
    MPI_Finalize();
    return 0;
}