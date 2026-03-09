#ifndef ROCKET_TRAJECTORY_H
#define ROCKET_TRAJECTORY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

// PHYSICAL CONSTANTS
#define G_CONST     6.67430e-11        // Gravitational constant (m^3 kg^-1 s^-2)
#define M_EARTH     5.972e24           // Mass of Earth (kg)
#define R_EARTH     6371000.0          // Radius of Earth (m)
#define G0          9.81               // Standard gravity at sea level (m/s^2)
#define RHO_0       1.225              // Air density at sea level (kg/m^3)
#define H_SCALE     8500.0             // Atmospheric scale height (m)
#define OMEGA_EARTH 7.2921e-5          // Earth's angular velocity (rad/s)
#define PI          3.14159265358979323846


// ROCKET PARAMETERS (Inspried by Falcon 9 engine specs with a mass-optimised structure)
#define INITIAL_MASS        549000.0    // Total initial mass (kg)
#define FUEL_MASS           522000.0    // Total fuel mass (kg)  [FIXED: was 418000]
#define DRY_MASS            27000.0     // Dry mass: structure + payload [FIXED: was 131000]
#define PAYLOAD_MASS        10000.0     // Payload mass (kg)
#define CROSS_SECTION_AREA  10.75       // Cross-sectional area (m^2)
#define DRAG_COEFFICIENT    0.3         // Drag coefficient [FIXED: reduced slightly]
#define ISP_SEA_LEVEL       282.0       // Specific impulse at sea level (s)
#define ISP_VACUUM          311.0       // Specific impulse in vacuum (s)
#define MAX_THRUST          7607000.0   // Maximum thrust (N) - 9 Merlin engines
#define BURN_TIME           190.0       // Actual burn time at full throttle (s) [FIXED]


// SIMULATION PARAMETERS
#define DT                  0.5         // Time step (seconds) - increased for stability
#define SIMULATION_TIME     1200.0       // Total simulation time (seconds)
#define TARGET_ALTITUDE     400000.0    // Target orbital altitude (m) - 400 km
#define TARGET_VELOCITY     7670.0      // Target orbital velocity (m/s)
#define MAX_ACCELERATION    50.0        // Max acceleration (m/s^2) - safety limit



// OPTIMIZATION PARAMETERS
#define NUM_CONTROL_POINTS  10          // Number of control points for pitch profile
#define POPULATION_SIZE     1000        // Number of trajectories to test
#define MIN_PITCH_ANGLE     0.0         // Minimum pitch angle (radians)
#define MAX_PITCH_ANGLE     (PI/2)      // Maximum pitch angle (90 degrees)


// LAUNCH PARAMETERS (Cape Canaveral)
#define LAUNCH_LATITUDE     28.5
#define LAUNCH_LATITUDE_RAD (LAUNCH_LATITUDE * PI / 180.0)


// FIXED RANDOM SEED for reproducible, fair comparison across all versions
#define FIXED_SEED  42






// DATA STRUCTURES
typedef struct {
    double x, y, z;        // Position (m)
    double vx, vy, vz;     // Velocity (m/s)
    double mass;            // Current mass (kg)
    double time;            // Current time (s)
} State;

typedef struct {
    double pitch_schedule[NUM_CONTROL_POINTS];
    double time_points[NUM_CONTROL_POINTS];
} ControlProfile;

typedef struct {
    double fuel_consumed;
    double final_altitude;
    double final_velocity;
    double max_acceleration;
    double max_dynamic_pressure;
    bool   reached_orbit;
    double fitness;
    ControlProfile control;
    int    trajectory_id;
} TrajectoryResult;

typedef struct {
    double thrust_x, thrust_y, thrust_z;
    double drag_x,   drag_y,   drag_z;
    double gravity_x,gravity_y,gravity_z;
} Forces;






// FUNCTION PROTOTYPES

// Physics
double calculate_air_density(double altitude);
double calculate_gravity(double r);
void   calculate_forces(State *state, double pitch, double throttle, Forces *forces);
void   runge_kutta_step(State *state, double pitch, double throttle);
double interpolate_pitch(ControlProfile *control, double time);

// Simulation
TrajectoryResult simulate_trajectory(ControlProfile *control, int id);
void   initialize_state(State *state);
double calculate_fitness(TrajectoryResult *result);

// Optimization
void   generate_random_control(ControlProfile *control);
void   initialize_population(ControlProfile *population, int size);
TrajectoryResult find_best_trajectory_serial(ControlProfile *population, int size);

// Utility
void   print_result(TrajectoryResult *result, const char *label);
void   save_results_to_file(TrajectoryResult *results, int count, const char *filename);
double get_wall_time();
void   print_performance_stats(const char *label, double execution_time, double speedup);
double calculate_energy(State *state);
double calculate_rmse(TrajectoryResult *ref, TrajectoryResult *test);



#endif // ROCKET_TRAJECTORY_H