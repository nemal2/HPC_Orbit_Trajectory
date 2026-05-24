/*
 * COMPARISON & ANALYSIS Program
 *
 * Runs Serial, OpenMP, MPI, Hybrid, and optionally CUDA.
 *
 * Standard build:
 *   mpicc -fopenmp -O3 -o compare compare.c rocket_physics.c rocket_utils.c -lm
 *   mpirun -np 4 ./compare
 *
 * CUDA-enabled build:
 *   make compare_cuda
 *   mpirun -np 4 ./compare_cuda
 *
 * If CUDA support is not compiled in, or if no CUDA device is available at
 * runtime, the CUDA row is skipped and the rest of the comparison still runs.
 */

#include "rocket_trajectory.h"
#include <mpi.h>
#include <omp.h>

#ifdef ENABLE_CUDA
extern TrajectoryResult find_best_trajectory_cuda_c(ControlProfile *population, int size);
extern int cuda_runtime_available_c(void);
#endif

#define N_OMP 3
#define N_MPI 3
#define N_HYB 3

static const char *orbit_text(TrajectoryResult *result)
{
    return result->reached_orbit ? "YES" : "NO";
}

static TrajectoryResult run_serial(ControlProfile *pop, int size)
{
    TrajectoryResult best;
    best.fitness = 1e9;

    for (int i = 0; i < size; i++) {
        TrajectoryResult r = simulate_trajectory(&pop[i], i);
        calculate_fitness(&r);
        if (r.fitness < best.fitness)
            best = r;
    }

    return best;
}

static TrajectoryResult run_openmp(ControlProfile *pop, int size, int nthreads)
{
    TrajectoryResult best;
    best.fitness = 1e9;

    omp_set_num_threads(nthreads);

#pragma omp parallel
    {
        TrajectoryResult local_best;
        local_best.fitness = 1e9;

#pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < size; i++) {
            TrajectoryResult r = simulate_trajectory(&pop[i], i);
            calculate_fitness(&r);
            if (r.fitness < local_best.fitness)
                local_best = r;
        }

#pragma omp critical
        {
            if (local_best.fitness < best.fitness)
                best = local_best;
        }
    }

    return best;
}

static int cuda_is_available(void)
{
#ifdef ENABLE_CUDA
    return cuda_runtime_available_c();
#else
    return 0;
#endif
}

static TrajectoryResult run_cuda(ControlProfile *pop, int size)
{
    TrajectoryResult best;
    best.fitness = 1e9;

#ifdef ENABLE_CUDA
    best = find_best_trajectory_cuda_c(pop, size);
#else
    (void)pop;
    (void)size;
#endif

    return best;
}

int main(int argc, char *argv[])
{
    int rank, nprocs, thread_support;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int max_threads = omp_get_max_threads();
    int omp_threads = (argc > 1) ? atoi(argv[1]) : max_threads;
    if (omp_threads < 1)
        omp_threads = max_threads;

    if (rank == 0) {
        printf("\n");
        printf("===============================================================\n");
        printf(" ROCKET TRAJECTORY OPTIMIZATION - COMPREHENSIVE ANALYSIS\n");
        printf(" Serial vs OpenMP vs MPI vs Hybrid vs CUDA\n");
        printf("===============================================================\n\n");
        printf("System Configuration:\n");
        printf("  MPI processes available:  %d\n", nprocs);
        printf("  OpenMP threads available: %d\n", max_threads);
        printf("  OpenMP threads selected:  %d\n", omp_threads);
        printf("  Fixed random seed:        %d\n", FIXED_SEED);
        printf("  Population size:          %d trajectories\n", POPULATION_SIZE);
#ifdef ENABLE_CUDA
        printf("  CUDA build:               enabled\n\n");
#else
        printf("  CUDA build:               disabled (build compare_cuda to enable)\n\n");
#endif
    }

    srand(FIXED_SEED);
    ControlProfile *pop = (ControlProfile *)malloc(POPULATION_SIZE * sizeof(ControlProfile));
    if (!pop) {
        if (rank == 0)
            fprintf(stderr, "Failed to allocate population.\n");
        MPI_Finalize();
        return 1;
    }
    initialize_population(pop, POPULATION_SIZE);

    int omp_threads_list[N_OMP] = {2, 4, 8};
    int mpi_procs_list[N_MPI] = {1, 2, 4};
    int hyb_p[N_HYB];
    int hyb_t[N_HYB];

    double serial_time = 0.0;
    double omp_times[N_OMP] = {0.0};
    double mpi_times[N_MPI] = {0.0};
    double hyb_times[N_HYB] = {0.0};
    double cuda_time = 0.0;

    TrajectoryResult serial_result;
    TrajectoryResult omp_results[N_OMP];
    TrajectoryResult mpi_results[N_MPI];
    TrajectoryResult hyb_results[N_HYB];
    TrajectoryResult cuda_result;

    serial_result.fitness = 1e9;
    cuda_result.fitness = 1e9;

    hyb_p[0] = 1;
    hyb_t[0] = omp_threads;
    hyb_p[1] = (nprocs >= 2) ? 2 : 1;
    hyb_t[1] = (omp_threads > 1) ? omp_threads / 2 : 1;
    hyb_p[2] = nprocs;
    hyb_t[2] = omp_threads;

    for (int t = 0; t < N_MPI; t++)
        if (mpi_procs_list[t] > nprocs)
            mpi_procs_list[t] = nprocs;

    if (rank == 0) {
        printf("==========================================================\n");
        printf("1. SERIAL BASELINE\n");
        printf("==========================================================\n");
        double t0 = MPI_Wtime();
        serial_result = run_serial(pop, POPULATION_SIZE);
        serial_time = MPI_Wtime() - t0;
        printf("  Time: %.3f s | Orbit: %s | Alt: %.1f km | Vel: %.1f m/s\n\n",
               serial_time, orbit_text(&serial_result),
               serial_result.final_altitude / 1000.0,
               serial_result.final_velocity);
    }
    MPI_Bcast(&serial_result, sizeof(TrajectoryResult), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial_time, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("==========================================================\n");
        printf("2. OPENMP (Shared Memory)\n");
        printf("==========================================================\n");
        for (int t = 0; t < N_OMP; t++) {
            double t0 = omp_get_wtime();
            omp_results[t] = run_openmp(pop, POPULATION_SIZE, omp_threads_list[t]);
            omp_times[t] = omp_get_wtime() - t0;
            printf("  %2d threads: %.3f s  speedup=%.2fx  orbit=%s\n",
                   omp_threads_list[t], omp_times[t],
                   serial_time / omp_times[t], orbit_text(&omp_results[t]));
        }
        printf("\n");
    }

    if (rank == 0) {
        printf("==========================================================\n");
        printf("3. MPI (Distributed Memory)\n");
        printf("==========================================================\n");
    }

    for (int t = 0; t < N_MPI; t++) {
        int use_p = mpi_procs_list[t];
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        TrajectoryResult global;
        global.fitness = 1e9;

        if (rank < use_p) {
            int per = POPULATION_SIZE / use_p;
            int start = rank * per;
            int end = (rank == use_p - 1) ? POPULATION_SIZE : start + per;

            TrajectoryResult local;
            local.fitness = 1e9;

            for (int i = start; i < end; i++) {
                TrajectoryResult r = simulate_trajectory(&pop[i], i);
                calculate_fitness(&r);
                if (r.fitness < local.fitness)
                    local = r;
            }

            if (rank == 0) {
                global = local;
                for (int i = 1; i < use_p; i++) {
                    TrajectoryResult tmp;
                    MPI_Recv(&tmp, sizeof(TrajectoryResult), MPI_BYTE,
                             i, 100 + t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (tmp.fitness < global.fitness)
                        global = tmp;
                }
            } else {
                MPI_Send(&local, sizeof(TrajectoryResult), MPI_BYTE,
                         0, 100 + t, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - t0;

        if (rank == 0) {
            mpi_times[t] = elapsed;
            mpi_results[t] = global;
            printf("  %2d process(es): %.3f s  speedup=%.2fx  orbit=%s\n",
                   use_p, elapsed, serial_time / elapsed, orbit_text(&global));
        }
        MPI_Bcast(&mpi_results[t], sizeof(TrajectoryResult), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&mpi_times[t], sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    if (rank == 0)
        printf("\n");

    if (rank == 0) {
        printf("==========================================================\n");
        printf("4. HYBRID MPI + OpenMP\n");
        printf("==========================================================\n");
    }

    for (int t = 0; t < N_HYB; t++) {
        int use_p = hyb_p[t];
        int use_t = hyb_t[t];
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        TrajectoryResult global;
        global.fitness = 1e9;

        if (rank < use_p) {
            int per = POPULATION_SIZE / use_p;
            int start = rank * per;
            int end = (rank == use_p - 1) ? POPULATION_SIZE : start + per;
            int count = end - start;

            TrajectoryResult local;
            local.fitness = 1e9;
            omp_set_num_threads(use_t);

#pragma omp parallel
            {
                TrajectoryResult local_best;
                local_best.fitness = 1e9;

#pragma omp for schedule(dynamic, 5)
                for (int i = 0; i < count; i++) {
                    TrajectoryResult r = simulate_trajectory(&pop[start + i], start + i);
                    calculate_fitness(&r);
                    if (r.fitness < local_best.fitness)
                        local_best = r;
                }

#pragma omp critical
                {
                    if (local_best.fitness < local.fitness)
                        local = local_best;
                }
            }

            if (rank == 0) {
                global = local;
                for (int i = 1; i < use_p; i++) {
                    TrajectoryResult tmp;
                    MPI_Recv(&tmp, sizeof(TrajectoryResult), MPI_BYTE,
                             i, 200 + t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (tmp.fitness < global.fitness)
                        global = tmp;
                }
            } else {
                MPI_Send(&local, sizeof(TrajectoryResult), MPI_BYTE,
                         0, 200 + t, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - t0;

        if (rank == 0) {
            hyb_times[t] = elapsed;
            hyb_results[t] = global;
            printf("  %dp x %dt = %2d workers: %.3f s  speedup=%.2fx  orbit=%s\n",
                   use_p, use_t, use_p * use_t, elapsed,
                   serial_time / elapsed, orbit_text(&global));
        }
        MPI_Bcast(&hyb_results[t], sizeof(TrajectoryResult), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&hyb_times[t], sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    int cuda_available = 0;
    if (rank == 0) {
        printf("\n==========================================================\n");
        printf("5. CUDA (GPU)\n");
        printf("==========================================================\n");
        cuda_available = cuda_is_available();
        if (cuda_available) {
            double t0 = MPI_Wtime();
            cuda_result = run_cuda(pop, POPULATION_SIZE);
            cuda_time = MPI_Wtime() - t0;
            printf("  CUDA GPU: %.3f s  speedup=%.2fx  orbit=%s\n",
                   cuda_time, serial_time / cuda_time, orbit_text(&cuda_result));
        } else {
#ifdef ENABLE_CUDA
            printf("  CUDA skipped: CUDA build is enabled, but no CUDA GPU was found.\n");
#else
            printf("  CUDA skipped: no CUDA build/device available on this computer.\n");
            printf("  To run CUDA on a CUDA computer, build with: make compare_cuda\n");
#endif
        }
    }
    MPI_Bcast(&cuda_available, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cuda_result, sizeof(TrajectoryResult), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cuda_time, sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n\n");
        printf("==========================================================================\n");
        printf(" COMPLETE PERFORMANCE & ACCURACY COMPARISON TABLE\n");
        printf("==========================================================================\n\n");

        printf("PERFORMANCE:\n");
        printf("---------------------------------------------------------------------\n");
        printf("%-28s %8s %9s %10s %10s %8s\n",
               "Version", "Workers", "Time(s)", "Speedup", "Efficiency", "Orbit?");
        printf("---------------------------------------------------------------------\n");

        printf("%-28s %8d %9.3f %10s %10s %8s\n",
               "Serial (baseline)", 1, serial_time, "1.00x", "100.0%",
               orbit_text(&serial_result));

        printf("  --- OpenMP --------------------------------------------------------\n");
        for (int t = 0; t < N_OMP; t++) {
            char label[40];
            snprintf(label, sizeof(label), "OpenMP (%d threads)", omp_threads_list[t]);
            double sp = serial_time / omp_times[t];
            double ef = (sp / omp_threads_list[t]) * 100.0;
            printf("%-28s %8d %9.3f %9.2fx %9.1f%% %8s\n",
                   label, omp_threads_list[t], omp_times[t], sp, ef,
                   orbit_text(&omp_results[t]));
        }

        printf("  --- MPI -----------------------------------------------------------\n");
        for (int t = 0; t < N_MPI; t++) {
            char label[40];
            snprintf(label, sizeof(label), "MPI (%d process%s)",
                     mpi_procs_list[t], mpi_procs_list[t] > 1 ? "es" : "");
            double sp = serial_time / mpi_times[t];
            double ef = (sp / mpi_procs_list[t]) * 100.0;
            printf("%-28s %8d %9.3f %9.2fx %9.1f%% %8s\n",
                   label, mpi_procs_list[t], mpi_times[t], sp, ef,
                   orbit_text(&mpi_results[t]));
        }

        printf("  --- Hybrid (MPI + OpenMP) ----------------------------------------\n");
        for (int t = 0; t < N_HYB; t++) {
            char label[40];
            snprintf(label, sizeof(label), "Hybrid (%dp x %dt)", hyb_p[t], hyb_t[t]);
            int workers = hyb_p[t] * hyb_t[t];
            double sp = serial_time / hyb_times[t];
            double ef = (sp / workers) * 100.0;
            printf("%-28s %8d %9.3f %9.2fx %9.1f%% %8s\n",
                   label, workers, hyb_times[t], sp, ef,
                   orbit_text(&hyb_results[t]));
        }

        printf("  --- CUDA ----------------------------------------------------------\n");
        if (cuda_available) {
            double sp = serial_time / cuda_time;
            printf("%-28s %8s %9.3f %9.2fx %10s %8s\n",
                   "CUDA (GPU)", "GPU", cuda_time, sp, "N/A", orbit_text(&cuda_result));
        } else {
            printf("%-28s %8s %9s %10s %10s %8s\n",
                   "CUDA (GPU)", "GPU", "SKIPPED", "N/A", "N/A", "N/A");
        }
        printf("---------------------------------------------------------------------\n\n");

        printf("ACCURACY vs SERIAL BASELINE:\n");
        printf("---------------------------------------------------------------------\n");
        printf("%-28s %10s %18s %8s\n", "Version", "RMSE", "Status", "Orbit?");
        printf("---------------------------------------------------------------------\n");

        for (int t = 0; t < N_OMP; t++) {
            char label[40];
            snprintf(label, sizeof(label), "OpenMP (%d threads)", omp_threads_list[t]);
            double rm = calculate_rmse(&serial_result, &omp_results[t]);
            printf("%-28s %10.4f %18s %8s\n", label, rm,
                   rm < 1.0 ? "Excellent" : rm < 10.0 ? "Acceptable" : "Differs",
                   orbit_text(&omp_results[t]));
        }
        for (int t = 0; t < N_MPI; t++) {
            char label[40];
            snprintf(label, sizeof(label), "MPI (%d process%s)",
                     mpi_procs_list[t], mpi_procs_list[t] > 1 ? "es" : "");
            double rm = calculate_rmse(&serial_result, &mpi_results[t]);
            printf("%-28s %10.4f %18s %8s\n", label, rm,
                   rm < 1.0 ? "Excellent" : rm < 10.0 ? "Acceptable" : "Differs",
                   orbit_text(&mpi_results[t]));
        }
        for (int t = 0; t < N_HYB; t++) {
            char label[40];
            snprintf(label, sizeof(label), "Hybrid (%dp x %dt)", hyb_p[t], hyb_t[t]);
            double rm = calculate_rmse(&serial_result, &hyb_results[t]);
            printf("%-28s %10.4f %18s %8s\n", label, rm,
                   rm < 1.0 ? "Excellent" : rm < 10.0 ? "Acceptable" : "Differs",
                   orbit_text(&hyb_results[t]));
        }
        if (cuda_available) {
            double rm = calculate_rmse(&serial_result, &cuda_result);
            printf("%-28s %10.4f %18s %8s\n", "CUDA (GPU)", rm,
                   rm < 1.0 ? "Excellent" : rm < 10.0 ? "Acceptable" : "Differs",
                   orbit_text(&cuda_result));
        } else {
            printf("%-28s %10s %18s %8s\n", "CUDA (GPU)", "N/A", "Skipped", "N/A");
        }
        printf("---------------------------------------------------------------------\n\n");

        double best_sp = 1.0;
        const char *best_ver = "Serial";
        for (int t = 0; t < N_OMP; t++) {
            double s = serial_time / omp_times[t];
            if (s > best_sp) {
                best_sp = s;
                best_ver = "OpenMP";
            }
        }
        for (int t = 0; t < N_MPI; t++) {
            double s = serial_time / mpi_times[t];
            if (s > best_sp) {
                best_sp = s;
                best_ver = "MPI";
            }
        }
        for (int t = 0; t < N_HYB; t++) {
            double s = serial_time / hyb_times[t];
            if (s > best_sp) {
                best_sp = s;
                best_ver = "Hybrid";
            }
        }
        if (cuda_available) {
            double s = serial_time / cuda_time;
            if (s > best_sp) {
                best_sp = s;
                best_ver = "CUDA";
            }
        }

        printf("KEY FINDINGS FOR REPORT:\n");
        printf("---------------------------------------------------------------------\n");
        printf("1. Best overall in this run: %s -> %.2fx speedup vs serial\n",
               best_ver, best_sp);
        printf("2. Parallelism changes execution time, not the optimization objective.\n");
        printf("3. OpenMP usually has the lowest single-node overhead.\n");
        printf("4. MPI and Hybrid are most useful when scaling across HPC nodes.\n");
        printf("5. CUDA is included only when the binary is built with CUDA and a CUDA\n");
        printf("   device is available on the computer running the comparison.\n");
        printf("6. Efficiency below 100%% is expected due to overhead and Amdahl's Law.\n");
        printf("---------------------------------------------------------------------\n\n");

        FILE *fp = fopen("performance_analysis.csv", "w");
        if (fp) {
            fprintf(fp, "Category,Version,Workers,Time_s,Speedup,Efficiency_pct,"
                        "RMSE,FuelKg,AltKm,VelMs,Fitness,Orbit\n");
            fprintf(fp, "Serial,Serial,1,%.3f,1.000,100.0,0.0000,%.2f,%.3f,%.2f,%.4f,%d\n",
                    serial_time, serial_result.fuel_consumed,
                    serial_result.final_altitude / 1000.0,
                    serial_result.final_velocity, serial_result.fitness,
                    serial_result.reached_orbit ? 1 : 0);
            for (int t = 0; t < N_OMP; t++) {
                double sp = serial_time / omp_times[t];
                double ef = (sp / omp_threads_list[t]) * 100.0;
                double rm = calculate_rmse(&serial_result, &omp_results[t]);
                fprintf(fp, "OpenMP,OpenMP_%dT,%d,%.3f,%.3f,%.1f,%.4f,%.2f,%.3f,%.2f,%.4f,%d\n",
                        omp_threads_list[t], omp_threads_list[t], omp_times[t], sp, ef, rm,
                        omp_results[t].fuel_consumed, omp_results[t].final_altitude / 1000.0,
                        omp_results[t].final_velocity, omp_results[t].fitness,
                        omp_results[t].reached_orbit ? 1 : 0);
            }
            for (int t = 0; t < N_MPI; t++) {
                double sp = serial_time / mpi_times[t];
                double ef = (sp / mpi_procs_list[t]) * 100.0;
                double rm = calculate_rmse(&serial_result, &mpi_results[t]);
                fprintf(fp, "MPI,MPI_%dP,%d,%.3f,%.3f,%.1f,%.4f,%.2f,%.3f,%.2f,%.4f,%d\n",
                        mpi_procs_list[t], mpi_procs_list[t], mpi_times[t], sp, ef, rm,
                        mpi_results[t].fuel_consumed, mpi_results[t].final_altitude / 1000.0,
                        mpi_results[t].final_velocity, mpi_results[t].fitness,
                        mpi_results[t].reached_orbit ? 1 : 0);
            }
            for (int t = 0; t < N_HYB; t++) {
                int workers = hyb_p[t] * hyb_t[t];
                double sp = serial_time / hyb_times[t];
                double ef = (sp / workers) * 100.0;
                double rm = calculate_rmse(&serial_result, &hyb_results[t]);
                fprintf(fp, "Hybrid,Hybrid_%dPx%dT,%d,%.3f,%.3f,%.1f,%.4f,%.2f,%.3f,%.2f,%.4f,%d\n",
                        hyb_p[t], hyb_t[t], workers, hyb_times[t], sp, ef, rm,
                        hyb_results[t].fuel_consumed, hyb_results[t].final_altitude / 1000.0,
                        hyb_results[t].final_velocity, hyb_results[t].fitness,
                        hyb_results[t].reached_orbit ? 1 : 0);
            }
            if (cuda_available) {
                double sp = serial_time / cuda_time;
                double rm = calculate_rmse(&serial_result, &cuda_result);
                fprintf(fp, "CUDA,CUDA_GPU,GPU,%.3f,%.3f,,%.4f,%.2f,%.3f,%.2f,%.4f,%d\n",
                        cuda_time, sp, rm, cuda_result.fuel_consumed,
                        cuda_result.final_altitude / 1000.0, cuda_result.final_velocity,
                        cuda_result.fitness, cuda_result.reached_orbit ? 1 : 0);
            } else {
                fprintf(fp, "CUDA,CUDA_GPU,GPU,SKIPPED,,,,,,,,\n");
            }
            fclose(fp);
            printf("Full results saved to performance_analysis.csv\n\n");
        }
    }

    free(pop);
    MPI_Finalize();
    return 0;
}
