#include "rocket_trajectory.h"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double random_double(double min, double max)
{
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

void generate_random_control(ControlProfile *control)
{
    for (int i = 0; i < NUM_CONTROL_POINTS; i++)
        control->time_points[i] =
            (BURN_TIME / (NUM_CONTROL_POINTS - 1)) * i;

    control->pitch_schedule[0] =
        random_double(PI * 0.46, PI * 0.50);

    for (int i = 1; i < NUM_CONTROL_POINTS; i++)
    {
        double fraction = (double)i / (NUM_CONTROL_POINTS - 1);
        double alpha = random_double(0.6, 1.4);

        double nominal =
            (PI / 2.0) * pow(1.0 - fraction, alpha);

        double noise = random_double(-0.08, 0.08);

        double angle = nominal + noise;

        if (angle < 0.0)
            angle = 0.0;

        if (angle > control->pitch_schedule[i - 1])
            angle = control->pitch_schedule[i - 1];

        control->pitch_schedule[i] = angle;
    }

    control->pitch_schedule[NUM_CONTROL_POINTS - 1] =
        random_double(0.0, PI * 0.028);
}

void initialize_population(ControlProfile *population, int size)
{
    for (int i = 0; i < size; i++)
        generate_random_control(&population[i]);
}

double get_wall_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);

    return (double)t.tv_sec +
           (double)t.tv_usec * 1e-6;
}

void print_result(TrajectoryResult *result, const char *label)
{
    printf("\n========================================\n");
    printf("%s\n", label);
    printf("========================================\n");

    printf("Trajectory ID: %d\n", result->trajectory_id);
    printf("Final Altitude: %.2f km\n",
           result->final_altitude / 1000.0);
    printf("Final Velocity: %.2f m/s\n",
           result->final_velocity);
    printf("Fuel Consumed: %.2f kg\n",
           result->fuel_consumed);
    printf("Fitness Score: %.4f\n",
           result->fitness);

    printf("Reached Orbit: %s\n",
           result->reached_orbit ? "YES ✓" : "NO ✗");
}