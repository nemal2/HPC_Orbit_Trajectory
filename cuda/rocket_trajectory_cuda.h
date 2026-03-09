#ifndef ROCKET_TRAJECTORY_CUDA_H
#define ROCKET_TRAJECTORY_CUDA_H

#include "rocket_trajectory.h"

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

HD double calculate_air_density(double altitude);
HD double calculate_gravity(double r);
HD void initialize_state(State *state);
HD double interpolate_pitch(ControlProfile *control, double time);
HD void runge_kutta_step(State *state, double pitch, double throttle);
HD TrajectoryResult simulate_trajectory(ControlProfile *control, int id);
HD double calculate_fitness(TrajectoryResult *result);

#endif