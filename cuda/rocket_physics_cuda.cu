/*
 * Rocket Trajectory Optimization - Physics Core
 */

#include "rocket_trajectory_cuda.h"
#include <string.h>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ============================================================================
// PHYSICS FUNCTIONS
// ============================================================================

/* Atmospheric density using exponential atmosphere model */
HD double calculate_air_density(double altitude)
{
    if (altitude < 0)
        return RHO_0;

    if (altitude > 150000)
        return 0.0;

    return RHO_0 * exp(-altitude / H_SCALE);
}

/* Gravity magnitude using Newton's law */
HD double calculate_gravity_mag(double r)
{
    if (r < R_EARTH)
        r = R_EARTH;

    return G_CONST * M_EARTH / (r * r);
}

// Compatibility wrapper
HD double calculate_gravity(double r)
{
    return calculate_gravity_mag(r);
}

/* Specific impulse varies with altitude */
HD static double calculate_isp(double altitude)
{
    if (altitude >= 50000)
        return ISP_VACUUM;

    if (altitude <= 0)
        return ISP_SEA_LEVEL;

    return ISP_SEA_LEVEL +
           (ISP_VACUUM - ISP_SEA_LEVEL) *
               (altitude / 50000.0);
}

/* Compute thrust, gravity, and drag forces */
HD void calculate_forces(State *state,
                         double pitch,
                         double throttle,
                         Forces *forces)
{
    double rx = state->x;
    double ry = state->y;
    double rz = state->z;

    double r = sqrt(rx * rx + ry * ry + rz * rz);

    double altitude = r - R_EARTH;

    /* Radial unit vector */
    double ur_x = rx / r;
    double ur_y = ry / r;
    double ur_z = rz / r;

    /* Tangential direction */
    double horiz_len =
        sqrt(rz * rz + rx * rx);

    double ut_x, ut_y, ut_z;

    if (horiz_len > 1.0)
    {
        ut_x = -rz / r;
        ut_y = 0.0;
        ut_z = rx / r;
    }
    else
    {
        ut_x = 1.0;
        ut_y = 0.0;
        ut_z = 0.0;
    }

    /* Gravity force */
    double g = calculate_gravity_mag(r);

    forces->gravity_x =
        -g * state->mass * ur_x;

    forces->gravity_y =
        -g * state->mass * ur_y;

    forces->gravity_z =
        -g * state->mass * ur_z;

    /* Rocket thrust */
    double isp =
        calculate_isp(altitude);

    double mass_flow =
        (MAX_THRUST * throttle) /
        (isp * G0);

    double thrust_mag =
        isp * G0 * mass_flow;

    double thrust_tang =
        thrust_mag * cos(pitch);

    double thrust_radial =
        thrust_mag * sin(pitch);

    forces->thrust_x =
        thrust_tang * ut_x +
        thrust_radial * ur_x;

    forces->thrust_y =
        thrust_tang * ut_y +
        thrust_radial * ur_y;

    forces->thrust_z =
        thrust_tang * ut_z +
        thrust_radial * ur_z;

    /* Aerodynamic drag */
    double vx = state->vx;
    double vy = state->vy;
    double vz = state->vz;

    double v =
        sqrt(vx * vx + vy * vy + vz * vz);

    double rho =
        calculate_air_density(altitude);

    double drag_mag =
        0.5 * rho * v * v *
        DRAG_COEFFICIENT *
        CROSS_SECTION_AREA;

    if (v > 0.001)
    {
        forces->drag_x = -drag_mag * vx / v;
        forces->drag_y = -drag_mag * vy / v;
        forces->drag_z = -drag_mag * vz / v;
    }
    else
    {
        forces->drag_x = 0.0;
        forces->drag_y = 0.0;
        forces->drag_z = 0.0;
    }
}

/* Linear interpolation of pitch schedule */
HD double interpolate_pitch(ControlProfile *control,
                            double time)
{
    if (time <= control->time_points[0])
        return control->pitch_schedule[0];

    if (time >= control->time_points[NUM_CONTROL_POINTS - 1])
        return control->pitch_schedule[NUM_CONTROL_POINTS - 1];

    for (int i = 0;
         i < NUM_CONTROL_POINTS - 1;
         i++)
    {
        if (time >= control->time_points[i] &&
            time <= control->time_points[i + 1])
        {
            double t0 = control->time_points[i];
            double t1 = control->time_points[i + 1];

            double p0 = control->pitch_schedule[i];
            double p1 = control->pitch_schedule[i + 1];

            double frac =
                (time - t0) / (t1 - t0);

            return p0 + (p1 - p0) * frac;
        }
    }

    return control->pitch_schedule[NUM_CONTROL_POINTS - 1];
}

/* Fourth-order Runge-Kutta integration */
HD void runge_kutta_step(State *state,
                         double pitch,
                         double throttle)
{
    Forces forces;
    State k[4], tmp;

#define DERIV(s, k_out)                                  \
    do                                                   \
    {                                                    \
        calculate_forces((s), pitch, throttle, &forces); \
                                                         \
        double tfx = forces.thrust_x +                   \
                     forces.drag_x +                     \
                     forces.gravity_x;                   \
                                                         \
        double tfy = forces.thrust_y +                   \
                     forces.drag_y +                     \
                     forces.gravity_y;                   \
                                                         \
        double tfz = forces.thrust_z +                   \
                     forces.drag_z +                     \
                     forces.gravity_z;                   \
                                                         \
        (k_out).x = (s)->vx;                             \
        (k_out).y = (s)->vy;                             \
        (k_out).z = (s)->vz;                             \
                                                         \
        (k_out).vx = tfx / (s)->mass;                    \
        (k_out).vy = tfy / (s)->mass;                    \
        (k_out).vz = tfz / (s)->mass;                    \
                                                         \
        double alt2 =                                    \
            sqrt((s)->x * (s)->x +                       \
                 (s)->y * (s)->y +                       \
                 (s)->z * (s)->z) -                      \
            R_EARTH;                                     \
                                                         \
        double isp2 = calculate_isp(alt2);               \
                                                         \
        double mf2 =                                     \
            (MAX_THRUST * throttle) /                    \
            (isp2 * G0);                                 \
                                                         \
        (k_out).mass = -(mf2 * throttle);                \
        (k_out).time = 1.0;                              \
    } while (0)

#define ADVANCE(base, kk, dt_scale, out)            \
    do                                              \
    {                                               \
        (out).x = (base)->x + dt_scale * (kk).x;    \
        (out).y = (base)->y + dt_scale * (kk).y;    \
        (out).z = (base)->z + dt_scale * (kk).z;    \
        (out).vx = (base)->vx + dt_scale * (kk).vx; \
        (out).vy = (base)->vy + dt_scale * (kk).vy; \
        (out).vz = (base)->vz + dt_scale * (kk).vz; \
        (out).mass = (base)->mass +                 \
                     dt_scale * (kk).mass;          \
    } while (0)

    DERIV(state, k[0]);

    ADVANCE(state, k[0], DT * 0.5, tmp);
    DERIV(&tmp, k[1]);

    ADVANCE(state, k[1], DT * 0.5, tmp);
    DERIV(&tmp, k[2]);

    ADVANCE(state, k[2], DT, tmp);
    DERIV(&tmp, k[3]);

#undef DERIV
#undef ADVANCE

    /* RK4 weighted update */
    state->x +=
        (DT / 6.0) *
        (k[0].x + 2 * k[1].x +
         2 * k[2].x + k[3].x);

    state->vx +=
        (DT / 6.0) *
        (k[0].vx + 2 * k[1].vx +
         2 * k[2].vx + k[3].vx);

    state->time += DT;

    if (state->mass < DRY_MASS)
        state->mass = DRY_MASS;
}

/* Initial launch state */
HD void initialize_state(State *state)
{
    state->x = R_EARTH;
    state->y = 0.0;
    state->z = 0.0;

    state->vx = 0.0;
    state->vy = 0.0;

    /* Earth rotational velocity boost */
    state->vz =
        OMEGA_EARTH *
        R_EARTH *
        cos(LAUNCH_LATITUDE_RAD);

    state->mass = INITIAL_MASS;
    state->time = 0.0;
}

/* Simulate one complete trajectory */
HD TrajectoryResult simulate_trajectory(ControlProfile *control,
                                        int id)
{
    State state;

    initialize_state(&state);

    double max_accel = 0.0;
    double max_q = 0.0;

    int num_steps =
        (int)(SIMULATION_TIME / DT);

    double best_orbit_score = 1e18;
    double best_alt = 0.0;
    double best_vel = 0.0;

    for (int step = 0;
         step < num_steps;
         step++)
    {
        double pitch =
            interpolate_pitch(control,
                              state.time);

        double throttle =
            (state.mass > DRY_MASS + 1.0)
                ? 1.0
                : 0.0;

        /* Numerical integration step */
        runge_kutta_step(&state,
                         pitch,
                         throttle);

        double r =
            sqrt(state.x * state.x +
                 state.y * state.y +
                 state.z * state.z);

        double alt = r - R_EARTH;

        /* Horizontal velocity */
        double vhoriz =
            sqrt(state.vx * state.vx +
                 state.vz * state.vz);

        /* Orbit matching score */
        double ae =
            (alt - TARGET_ALTITUDE) /
            TARGET_ALTITUDE;

        double ve =
            (vhoriz - TARGET_VELOCITY) /
            TARGET_VELOCITY;

        double score =
            ae * ae + ve * ve;

        if (score < best_orbit_score)
        {
            best_orbit_score = score;
            best_alt = alt;
            best_vel = vhoriz;
        }
    }

    TrajectoryResult result;

    result.fuel_consumed =
        INITIAL_MASS - state.mass;

    result.final_altitude = best_alt;
    result.final_velocity = best_vel;

    result.trajectory_id = id;

    double alt_err =
        fabs(best_alt - TARGET_ALTITUDE) /
        TARGET_ALTITUDE;

    double vel_err =
        fabs(best_vel - TARGET_VELOCITY) /
        TARGET_VELOCITY;

    result.reached_orbit =
        (alt_err < 0.15 &&
         vel_err < 0.15);

    memcpy(&result.control,
           control,
           sizeof(ControlProfile));

    result.fitness = 0.0;

    return result;
}

/* Fitness evaluation */
HD double calculate_fitness(TrajectoryResult *result)
{
    double fitness = 0.0;

    double alt_err =
        fabs(result->final_altitude -
             TARGET_ALTITUDE);

    double vel_err =
        fabs(result->final_velocity -
             TARGET_VELOCITY);

    /* Altitude penalty */
    fitness +=
        (alt_err / TARGET_ALTITUDE) *
        500.0;

    /* Velocity penalty */
    fitness +=
        (vel_err / TARGET_VELOCITY) *
        500.0;

    /* Fuel penalty */
    fitness +=
        result->fuel_consumed /
        FUEL_MASS * 50.0;

    /* Orbit reward */
    if (result->reached_orbit)
        fitness -= 400.0;

    result->fitness = fitness;

    return fitness;
}

/* Mechanical energy calculation */
HD double calculate_energy(State *state)
{
    double r =
        sqrt(state->x * state->x +
             state->y * state->y +
             state->z * state->z);

    double v2 =
        state->vx * state->vx +
        state->vy * state->vy +
        state->vz * state->vz;

    return 0.5 * state->mass * v2 - G_CONST * M_EARTH *
                                        state->mass / r;
}

/* RMSE comparison between implementations */
HD double calculate_rmse(TrajectoryResult *ref,
                         TrajectoryResult *test)
{
    double d1 =
        (ref->fuel_consumed -
         test->fuel_consumed) /
        1000.0;

    double d2 =
        (ref->final_altitude -
         test->final_altitude) /
        1000.0;

    double d3 =
        (ref->final_velocity -
         test->final_velocity) /
        10.0;

    return sqrt(
        (d1 * d1 +
         d2 * d2 +
         d3 * d3) /
        3.0);
}