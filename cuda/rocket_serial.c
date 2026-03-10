//SERIAL VERSION - Rocket Trajectory Optimization

// pseudocode

//== 1. Initialize random population of trajectories
//== 2. For each trajectory
//==       simulate rocket flight
//==       calculate physics
//==       evaluate fitness
//== 3. Compare all trajectories
//== 4. Select the best trajectory
//== 5. Print results

#include "rocket_trajectory.h"

int main(int argc, char *argv[]) {
    printf("\n");
    printf("  ROCKET TRAJECTORY OPTIMIZATION - SERIAL VERSION          \n");
    printf("   SpaceX-Inspired Trajectory Optimization                  \n");
  printf("\n");printf("\n");

    // Fixed seed for fair comparison - ALL versions use this same seed
    srand(FIXED_SEED);


    printf("Initializing population of %d trajectories (seed=%d)...\n",
           POPULATION_SIZE, FIXED_SEED);
    ControlProfile *population = (ControlProfile*)malloc(POPULATION_SIZE * sizeof(ControlProfile));
    initialize_population(population, POPULATION_SIZE);
    printf("Population initialized\n\n");



    printf("========================================\n");
    printf("SIMULATION PARAMETERS\n");
    printf("========================================\n");
    printf("Target Altitude:     %.0f km\n", TARGET_ALTITUDE / 1000.0);
    printf("Target Velocity:     %.0f m/s (horizontal)\n", TARGET_VELOCITY);
    printf("Population Size:     %d\n", POPULATION_SIZE);
    printf("Control Points:      %d\n", NUM_CONTROL_POINTS);
    printf("Time Step:           %.2f s\n", DT);
    printf("Max Simulation Time: %.0f s\n", SIMULATION_TIME);
    printf("Fixed Random Seed:   %d\n", FIXED_SEED);
    printf("========================================\n\n");

    printf("Starting SERIAL optimization...\n");
    printf("Testing %d trajectories sequentially...\n\n", POPULATION_SIZE);




    double start_time = get_wall_time();
    TrajectoryResult best = find_best_trajectory_serial(population, POPULATION_SIZE);
    double end_time   = get_wall_time();
    double exec_time  = end_time - start_time;

    print_result(&best, "BEST TRAJECTORY FOUND (SERIAL)");
    print_performance_stats("Serial Execution", exec_time, 0.0);

    TrajectoryResult results[1] = {best};
    save_results_to_file(results, 1, "results_serial.csv");

    // Verification
    State  init_state;
    initialize_state(&init_state);
    double init_energy = calculate_energy(&init_state);

    printf("\n========================================\n");
    printf("VERIFICATION\n");
    printf("========================================\n");
    printf("Initial Energy:  %.2e J\n", init_energy);
    printf("Orbit Reached:   %s\n", best.reached_orbit ? "YES " : "NO ");






    double alt_err_pct = fabs(best.final_altitude - TARGET_ALTITUDE) / TARGET_ALTITUDE * 100.0;
    double vel_err_pct = fabs(best.final_velocity  - TARGET_VELOCITY)  / TARGET_VELOCITY  * 100.0;
    printf("Altitude Error:  %.2f%%\n", alt_err_pct);
    printf("Velocity Error:  %.2f%%\n", vel_err_pct);

    if (alt_err_pct < 10.0 && vel_err_pct < 10.0)
        printf("Trajectory is within acceptable tolerances!\n");
    else
        printf("Trajectory needs improvement (try larger population)\n");
    printf("========================================\n\n");

    free(population);
    printf("  Serial optimization complete!\n");
    printf("  Results saved to results_serial.csv\n");
    printf("  Execution time: %.3f seconds\n\n", exec_time);
    return 0;


   
}