/*
 * HYBRID VERSION - Rocket Trajectory Optimization (FIXED)
 * Combines MPI (distributed memory) + OpenMP (shared memory)
 *
 * FIXES vs original:
 *   1. Fixed seed: every process uses FIXED_SEED → same population, fair comparison
 *   2. Garbled output fixed: only process 0, thread 0 prints progress
 *   3. print_result called once, outside parallel region, inside rank==0 block
 *   4. You MUST run with mpirun: "mpirun -np 4 ./rocket_hybrid 4"
 *      Running ./rocket_hybrid directly only gives 1 MPI process (no MPI speedup)
 *
 * Compile: mpicc -fopenmp -o rocket_hybrid rocket_hybrid.c rocket_physics.c rocket_utils.c -lm -O3
 * Run:     mpirun -np 4 ./rocket_hybrid 4    (4 procs × 4 threads = 16 workers)
 *          mpirun -np 2 ./rocket_hybrid 8    (2 procs × 8 threads = 16 workers)
 */

#include "rocket_trajectory.h"
#include <mpi.h>
#include <omp.h>

TrajectoryResult find_best_trajectory_hybrid(ControlProfile *population, int size,
                                             int rank, int num_procs, int num_threads) {
    TrajectoryResult local_best, global_best;
    local_best.fitness  = 1e9;
    global_best.fitness = 1e9;

    int per_proc    = size / num_procs;
    int start_idx   = rank * per_proc;
    int end_idx     = (rank == num_procs-1) ? size : (rank+1) * per_proc;
    int local_count = end_idx - start_idx;

    if (rank == 0) {
        printf("  HYBRID Configuration:\n");
        printf("    MPI processes:           %d\n", num_procs);
        printf("    OpenMP threads/process:  %d\n", num_threads);
        printf("    Total parallel workers:  %d\n", num_procs * num_threads);
        printf("    Trajectories per proc:   ~%d\n", per_proc);
        printf("    Trajectories per thread: ~%d\n\n", per_proc / num_threads);
    }

    // Same fix as MPI: all processes regenerate identical population
    if (rank != 0) {
        srand(FIXED_SEED);
        initialize_population(population, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        printf("  Process %d: Testing trajectories %d-%d with %d threads\n",
               rank, start_idx, end_idx-1, num_threads);

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        TrajectoryResult thread_best;
        thread_best.fitness = 1e9;

        #pragma omp for schedule(dynamic, 5)
        for (int i = 0; i < local_count; i++) {
            int gid = start_idx + i;
            TrajectoryResult result = simulate_trajectory(&population[gid], gid);
            calculate_fitness(&result);

            if (result.fitness < thread_best.fitness)
                thread_best = result;
        }

        // FIX: only rank 0, thread 0 prints progress to avoid garbled output
        #pragma omp critical
        {
            if (thread_best.fitness < local_best.fitness) {
                local_best = thread_best;
                if (rank == 0)
                    printf("  Process 0, Thread %d: New local best fitness = %.4f\n",
                           tid, thread_best.fitness);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("  Process 0: Local best fitness = %.4f (after OpenMP reduction)\n\n",
               local_best.fitness);

    // Gather best from all MPI processes
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
    int rank, num_procs, thread_support;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    if (num_threads < 1) num_threads = omp_get_max_threads();

    if (rank == 0) {
        printf("\n");
        printf("╔════════════════════════════════════════════════════════════╗\n");
        printf("║   ROCKET TRAJECTORY OPTIMIZATION - HYBRID VERSION (FIXED)  ║\n");
        printf("║   MPI (Distributed) + OpenMP (Shared Memory)               ║\n");
        printf("╚════════════════════════════════════════════════════════════╝\n\n");

        if (thread_support < MPI_THREAD_FUNNELED)
            printf("WARNING: MPI thread support level is low (%d). Results may vary.\n\n",
                   thread_support);

        printf("Hybrid Configuration:\n");
        printf("  MPI processes:         %d\n",       num_procs);
        printf("  OpenMP threads/proc:   %d\n",       num_threads);
        printf("  Total workers:         %d\n",       num_procs * num_threads);
        printf("  Fixed Random Seed:     %d\n\n",     FIXED_SEED);

        if (num_procs == 1)
            printf("NOTE: Only 1 MPI process detected. For full hybrid speedup,\n"
                   "      run with: mpirun -np 4 ./rocket_hybrid 4\n\n");
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
        printf("OpenMP Threads:      %d per process\n", num_threads);
        printf("Total Parallelism:   %d workers\n", num_procs * num_threads);
        printf("========================================\n\n");
        printf("Starting HYBRID (MPI+OpenMP) optimization...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    TrajectoryResult best = find_best_trajectory_hybrid(population, POPULATION_SIZE,
                                                        rank, num_procs, num_threads);

    MPI_Barrier(MPI_COMM_WORLD);
    double exec_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        // FIX: print_result called ONCE, outside parallel region → clean output
        print_result(&best, "BEST TRAJECTORY FOUND (HYBRID MPI+OpenMP)");

        int total_workers = num_procs * num_threads;
        printf("\n========================================\n");
        printf("PERFORMANCE: HYBRID (%d procs x %d threads)\n", num_procs, num_threads);
        printf("========================================\n");
        printf("Total parallel workers:  %d\n",   total_workers);
        printf("Execution Time:          %.3f seconds\n", exec_time);
        printf("Trajectories/second:     %.1f\n", POPULATION_SIZE / exec_time);
        printf("Theoretical speedup:     %.2fx (with %d workers)\n",
               (double)total_workers, total_workers);
        printf("========================================\n\n");

        TrajectoryResult results[1] = {best};
        save_results_to_file(results, 1, "results_hybrid.csv");

        printf("========================================\n");
        printf("VERIFICATION\n");
        printf("========================================\n");
        printf("Orbit Reached:  %s\n", best.reached_orbit ? "YES ✓" : "NO ✗");
        double a = fabs(best.final_altitude - TARGET_ALTITUDE) / TARGET_ALTITUDE * 100.0;
        double v = fabs(best.final_velocity  - TARGET_VELOCITY)  / TARGET_VELOCITY  * 100.0;
        printf("Altitude Error: %.2f%%\n", a);
        printf("Velocity Error: %.2f%%\n", v);
        if (a < 10.0 && v < 10.0)
            printf("✓ Trajectory is within acceptable tolerances!\n");
        printf("========================================\n\n");

        printf("✓ HYBRID optimization complete!\n");
        printf("  Results saved to results_hybrid.csv\n");
        printf("  %d processes × %d threads = %d workers, %.3f seconds\n\n",
               num_procs, num_threads, total_workers, exec_time);
    }

    free(population);
    MPI_Finalize();
    return 0;
}