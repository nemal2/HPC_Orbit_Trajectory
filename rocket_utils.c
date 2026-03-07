// Rocket Trajectory Optimization - Utility Functions 

#include "rocket_trajectory.h"
#include <sys/time.h>


// RANDOM NUMBER HELPERS
static double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}


/* Typical good Falcon-9-like profile:
 *   0 s  : ~88° (near vertical, clears launch tower)
 *   ~10 s: start gravity turn
 *   ~120s: ~45°
 *   ~300s: ~10°
 *   ~550s: ~2° (nearly horizontal at MECO)
 */



 void generate_random_control(ControlProfile *control) {
    // Evenly-spaced time points across the burn
    for (int i = 0; i < NUM_CONTROL_POINTS; i++)
        control->time_points[i] = (BURN_TIME / (NUM_CONTROL_POINTS - 1)) * i;

    // Start nearly vertical
    control->pitch_schedule[0] = random_double(PI*0.46, PI*0.50);  // 83-90°

    // Gravity-turn: angle decreases roughly as a half-cosine
    // Add randomness so we explore the space
    for (int i = 1; i < NUM_CONTROL_POINTS; i++) {
        double fraction  = (double)i / (NUM_CONTROL_POINTS - 1);   // 0 → 1
        // Nominal profile: PI/2 * (1-fraction)^alpha, alpha randomised per trajectory
        double alpha     = random_double(0.6, 1.4);
        double nominal   = (PI / 2.0) * pow(1.0 - fraction, alpha);
        double noise     = random_double(-0.08, 0.08);
        double angle     = nominal + noise;
        // Clamp to [0, previous value] so pitch is monotonically decreasing
        if (angle < 0.0) angle = 0.0;
        if (angle > control->pitch_schedule[i-1]) angle = control->pitch_schedule[i-1];
        control->pitch_schedule[i] = angle;
    }

    // Ensure final point is nearly horizontal (0–5°)
    control->pitch_schedule[NUM_CONTROL_POINTS-1] = random_double(0.0, PI*0.028);
}



void initialize_population(ControlProfile *population, int size) {
    for (int i = 0; i < size; i++)
        generate_random_control(&population[i]);
}



// SERIAL FIND BEST
TrajectoryResult find_best_trajectory_serial(ControlProfile *population, int size) {
    TrajectoryResult best;
    best.fitness = 1e9;

    for (int i = 0; i < size; i++) {
        TrajectoryResult result = simulate_trajectory(&population[i], i);
        calculate_fitness(&result);

        if (result.fitness < best.fitness)
            best = result;

        if ((i + 1) % 100 == 0)
            printf("  Tested %d/%d trajectories...\n", i + 1, size);
    }
    return best;
}



// UTILITY
double get_wall_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}


void print_result(TrajectoryResult *result, const char *label) {
    printf("\n========================================\n");
    printf("%s\n", label);
    printf("========================================\n");
    printf("Trajectory ID:        %d\n", result->trajectory_id);
    printf("Fuel Consumed:        %.2f kg (%.1f%%)\n",
           result->fuel_consumed,
           (result->fuel_consumed / FUEL_MASS) * 100.0);
    printf("Final Altitude:       %.2f km\n",   result->final_altitude / 1000.0);
    printf("Target Altitude:      %.2f km\n",   TARGET_ALTITUDE / 1000.0);
    printf("Altitude Error:       %.2f km\n",
           fabs(result->final_altitude - TARGET_ALTITUDE) / 1000.0);
    printf("Final Horiz Velocity: %.2f m/s\n",  result->final_velocity);
    printf("Target Velocity:      %.2f m/s\n",  TARGET_VELOCITY);
    printf("Velocity Error:       %.2f m/s\n",
           fabs(result->final_velocity - TARGET_VELOCITY));
    printf("Max Acceleration:     %.2f m/s^2 (%.2f g)\n",
           result->max_acceleration,
           result->max_acceleration / G0);
    printf("Max Dynamic Pressure: %.2f kPa\n",  result->max_dynamic_pressure / 1000.0);
    printf("Reached Orbit:        %s\n",         result->reached_orbit ? "YES ✓" : "NO ✗");
    printf("Fitness Score:        %.4f\n",        result->fitness);
    printf("========================================\n");

    printf("\nPitch Schedule:\n");
    for (int i = 0; i < NUM_CONTROL_POINTS; i++)
        printf("  t=%5.1fs: %5.1f deg\n",
               result->control.time_points[i],
               result->control.pitch_schedule[i] * 180.0 / PI);
    printf("\n");
}



void save_results_to_file(TrajectoryResult *results, int count, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) { printf("Error: Cannot open %s\n", filename); return; }

    fprintf(fp, "ID,FuelConsumed,FinalAltitude_km,FinalVelocity_ms,MaxAccel,MaxQ_kPa,ReachedOrbit,Fitness\n");
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%d,%.2f,%.3f,%.2f,%.2f,%.2f,%d,%.4f\n",
                results[i].trajectory_id,
                results[i].fuel_consumed,
                results[i].final_altitude / 1000.0,
                results[i].final_velocity,
                results[i].max_acceleration,
                results[i].max_dynamic_pressure / 1000.0,
                results[i].reached_orbit ? 1 : 0,
                results[i].fitness);
    }
    fclose(fp);
    printf("Results saved to %s\n", filename);
}


void print_performance_stats(const char *label, double exec_time, double speedup) {
    printf("\n========================================\n");
    printf("PERFORMANCE: %s\n", label);
    printf("========================================\n");
    printf("Execution Time:  %.3f seconds\n", exec_time);
    if (speedup > 0.0)
        printf("Speedup:         %.2fx\n", speedup);
    printf("========================================\n\n");
}
