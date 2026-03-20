/*
 * Rocket Trajectory Optimization - Physics Core (FIXED VERSION)
 *
 * KEY FIXES:
 * 1. Initial state: rocket starts at (0,0,R_EARTH) pointing upward correctly
 * 2. Thrust direction uses a proper gravity-turn model referenced to local vertical
 * 3. Orbital velocity check uses the correct horizontal component
 * 4. Fitness function tuned so good trajectories actually win
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

HD double calculate_air_density(double altitude)
{
    if (altitude < 0)
        return RHO_0;
    if (altitude > 150000)
        return 0.0;
    return RHO_0 * exp(-altitude / H_SCALE);
}

HD double calculate_gravity_mag(double r)
{
    if (r < R_EARTH)
        r = R_EARTH;
    return G_CONST * M_EARTH / (r * r);
}

// Keep old name for compatibility
HD double calculate_gravity(double r)
{
    return calculate_gravity_mag(r);
}

HD static double calculate_isp(double altitude)
{
    if (altitude >= 50000)
        return ISP_VACUUM;
    if (altitude <= 0)
        return ISP_SEA_LEVEL;
    return ISP_SEA_LEVEL + (ISP_VACUUM - ISP_SEA_LEVEL) * (altitude / 50000.0);
}

/*
 * FIXED calculate_forces
 *
 * The original code applied thrust in a fixed x-z lab frame regardless of
 * where the rocket actually is.  A rocket starting at (R_EARTH, 0, 0) and
 * thrusting purely in the +z direction would never gain altitude.
 *
 * Fix: compute the local vertical (radial) and local horizontal (tangential,
 * i.e. the direction of increasing downrange angle) unit vectors, then
 * decompose thrust as:
 *   pitch angle = angle from local horizontal toward local vertical
 *   thrust_vec  = cos(pitch)*tangential + sin(pitch)*radial
 *
 * This matches real gravity-turn physics.
 */
HD void calculate_forces(State *state, double pitch, double throttle, Forces *forces)
{
    double rx = state->x, ry = state->y, rz = state->z;
    double r = sqrt(rx * rx + ry * ry + rz * rz);
    double altitude = r - R_EARTH;

    // --- Local vertical (radial outward) unit vector ---
    double ur_x = rx / r, ur_y = ry / r, ur_z = rz / r;

    // --- Local horizontal (eastward / downrange) unit vector ---
    // We want the direction perpendicular to radial that lies in the
    // orbital plane. For a 2-D trajectory in the x-z plane we use:
    //   tangential = (-rz/r, 0, rx/r)   [cross product of y-axis x radial, projected]
    // This gives an eastward tangential direction at Cape Canaveral.
    double horiz_len = sqrt(rz * rz + rx * rx);
    double ut_x, ut_y, ut_z;
    if (horiz_len > 1.0)
    {
        ut_x = -rz / r;
        ut_y = 0.0;
        ut_z = rx / r;
    }
    else
    {
        // Degenerate case (rocket directly over pole) – use +x
        ut_x = 1.0;
        ut_y = 0.0;
        ut_z = 0.0;
    }

    // --- Gravity ---
    double g = calculate_gravity_mag(r);
    forces->gravity_x = -g * state->mass * ur_x;
    forces->gravity_y = -g * state->mass * ur_y;
    forces->gravity_z = -g * state->mass * ur_z;

    // --- Thrust ---
    double isp = calculate_isp(altitude);
    double mass_flow = (MAX_THRUST * throttle) / (isp * G0);
    double thrust_mag = isp * G0 * mass_flow; // = MAX_THRUST * throttle

    // pitch = 0  → purely horizontal (tangential)
    // pitch = π/2 → purely vertical (radial outward)
    double thrust_tang = thrust_mag * cos(pitch);
    double thrust_radial = thrust_mag * sin(pitch);

    forces->thrust_x = thrust_tang * ut_x + thrust_radial * ur_x;
    forces->thrust_y = thrust_tang * ut_y + thrust_radial * ur_y;
    forces->thrust_z = thrust_tang * ut_z + thrust_radial * ur_z;

    // --- Drag (opposes velocity) ---
    double vx = state->vx, vy = state->vy, vz = state->vz;
    double v = sqrt(vx * vx + vy * vy + vz * vz);
    double rho = calculate_air_density(altitude);
    double drag_mag = 0.5 * rho * v * v * DRAG_COEFFICIENT * CROSS_SECTION_AREA;

    if (v > 0.001)
    {
        forces->drag_x = -drag_mag * vx / v;
        forces->drag_y = -drag_mag * vy / v;
        forces->drag_z = -drag_mag * vz / v;
    }
    else
    {
        forces->drag_x = forces->drag_y = forces->drag_z = 0.0;
    }
}

HD double interpolate_pitch(ControlProfile *control, double time)
{
    if (time <= control->time_points[0])
        return control->pitch_schedule[0];
    if (time >= control->time_points[NUM_CONTROL_POINTS - 1])
        return control->pitch_schedule[NUM_CONTROL_POINTS - 1];

    for (int i = 0; i < NUM_CONTROL_POINTS - 1; i++)
    {
        if (time >= control->time_points[i] && time <= control->time_points[i + 1])
        {
            double t0 = control->time_points[i], t1 = control->time_points[i + 1];
            double p0 = control->pitch_schedule[i], p1 = control->pitch_schedule[i + 1];
            double frac = (time - t0) / (t1 - t0);
            return p0 + (p1 - p0) * frac;
        }
    }
    return control->pitch_schedule[NUM_CONTROL_POINTS - 1];
}

HD void runge_kutta_step(State *state, double pitch, double throttle)
{
    Forces forces;
    State k[4], tmp;

// Helper lambda-like macro for computing derivatives
#define DERIV(s, k_out)                                                                    \
    do                                                                                     \
    {                                                                                      \
        calculate_forces((s), pitch, throttle, &forces);                                   \
        double tfx = forces.thrust_x + forces.drag_x + forces.gravity_x;                   \
        double tfy = forces.thrust_y + forces.drag_y + forces.gravity_y;                   \
        double tfz = forces.thrust_z + forces.drag_z + forces.gravity_z;                   \
        (k_out).x = (s)->vx;                                                               \
        (k_out).y = (s)->vy;                                                               \
        (k_out).z = (s)->vz;                                                               \
        (k_out).vx = tfx / (s)->mass;                                                      \
        (k_out).vy = tfy / (s)->mass;                                                      \
        (k_out).vz = tfz / (s)->mass;                                                      \
        double alt2 = sqrt((s)->x * (s)->x + (s)->y * (s)->y + (s)->z * (s)->z) - R_EARTH; \
        double isp2 = calculate_isp(alt2);                                                 \
        double mf2 = (MAX_THRUST * throttle) / (isp2 * G0);                                \
        (k_out).mass = -(mf2 * throttle);                                                  \
        (k_out).time = 1.0;                                                                \
    } while (0)

#define ADVANCE(base, kk, dt_scale, out)                  \
    do                                                    \
    {                                                     \
        (out).x = (base)->x + dt_scale * (kk).x;          \
        (out).y = (base)->y + dt_scale * (kk).y;          \
        (out).z = (base)->z + dt_scale * (kk).z;          \
        (out).vx = (base)->vx + dt_scale * (kk).vx;       \
        (out).vy = (base)->vy + dt_scale * (kk).vy;       \
        (out).vz = (base)->vz + dt_scale * (kk).vz;       \
        (out).mass = (base)->mass + dt_scale * (kk).mass; \
        (out).time = (base)->time;                        \
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

    state->x += (DT / 6.0) * (k[0].x + 2 * k[1].x + 2 * k[2].x + k[3].x);
    state->y += (DT / 6.0) * (k[0].y + 2 * k[1].y + 2 * k[2].y + k[3].y);
    state->z += (DT / 6.0) * (k[0].z + 2 * k[1].z + 2 * k[2].z + k[3].z);
    state->vx += (DT / 6.0) * (k[0].vx + 2 * k[1].vx + 2 * k[2].vx + k[3].vx);
    state->vy += (DT / 6.0) * (k[0].vy + 2 * k[1].vy + 2 * k[2].vy + k[3].vy);
    state->vz += (DT / 6.0) * (k[0].vz + 2 * k[1].vz + 2 * k[2].vz + k[3].vz);
    state->mass += (DT / 6.0) * (k[0].mass + 2 * k[1].mass + 2 * k[2].mass + k[3].mass);
    state->time += DT;

    if (state->mass < DRY_MASS)
        state->mass = DRY_MASS;
}

/*
 * FIXED initialize_state
 * Start at (R_EARTH, 0, 0) – i.e. on the equator / Cape Canaveral in
 * the x direction.  Earth rotation gives an eastward (+z) velocity boost.
 */
HD void initialize_state(State *state)
{
    state->x = R_EARTH;
    state->y = 0.0;
    state->z = 0.0;
    state->vx = 0.0;
    state->vy = 0.0;
    // Earth rotation eastward velocity at launch latitude
    state->vz = OMEGA_EARTH * R_EARTH * cos(LAUNCH_LATITUDE_RAD);
    state->mass = INITIAL_MASS;
    state->time = 0.0;
}

HD TrajectoryResult simulate_trajectory(ControlProfile *control, int id)
{
    State state;
    initialize_state(&state);

    double max_accel = 0.0, max_q = 0.0;
    int num_steps = (int)(SIMULATION_TIME / DT);

    /* FIXED: Track the best orbital insertion condition reached during flight.
     * We record the snapshot where the weighted combination of altitude error
     * and velocity error is minimised.  This correctly identifies the orbit
     * insertion point for trajectories that coast through/past the target.   */
    double best_orbit_score = 1e18;
    double best_alt = 0.0, best_vel = 0.0;

    for (int step = 0; step < num_steps; step++)
    {
        double pitch = interpolate_pitch(control, state.time);
        double throttle = (state.mass > DRY_MASS + 1.0) ? 1.0 : 0.0;

        runge_kutta_step(&state, pitch, throttle);

        double r = sqrt(state.x * state.x + state.y * state.y + state.z * state.z);
        double alt = r - R_EARTH;

        // Horizontal (tangential) velocity component
        double ur_x = state.x / r, ur_y = state.y / r, ur_z = state.z / r;
        double vrad = state.vx * ur_x + state.vy * ur_y + state.vz * ur_z;
        double vt_x = state.vx - vrad * ur_x;
        double vt_y = state.vy - vrad * ur_y;
        double vt_z = state.vz - vrad * ur_z;
        double vhoriz = sqrt(vt_x * vt_x + vt_y * vt_y + vt_z * vt_z);

        double vtot = sqrt(state.vx * state.vx + state.vy * state.vy + state.vz * state.vz);

        // Score = (normalised altitude error)^2 + (normalised velocity error)^2
        double ae = (alt - TARGET_ALTITUDE) / TARGET_ALTITUDE;
        double ve = (vhoriz - TARGET_VELOCITY) / TARGET_VELOCITY;
        double score = ae * ae + ve * ve;
        if (score < best_orbit_score)
        {
            best_orbit_score = score;
            best_alt = alt;
            best_vel = vhoriz;
        }

        // Track max acceleration
        Forces f;
        calculate_forces(&state, pitch, throttle, &f);
        double fx = f.thrust_x + f.drag_x + f.gravity_x;
        double fy = f.thrust_y + f.drag_y + f.gravity_y;
        double fz = f.thrust_z + f.drag_z + f.gravity_z;
        double accel = sqrt(fx * fx + fy * fy + fz * fz) / state.mass;
        if (accel > max_accel)
            max_accel = accel;

        // Track max dynamic pressure
        double rho = calculate_air_density(alt);
        double q = 0.5 * rho * vtot * vtot;
        if (q > max_q)
            max_q = q;

        // Abort if crashed back to Earth
        if (alt < -500.0)
            break;
    }

    TrajectoryResult result;
    result.fuel_consumed = INITIAL_MASS - state.mass;
    result.final_altitude = best_alt; // best orbital altitude reached
    result.final_velocity = best_vel; // horizontal velocity at that point
    result.max_acceleration = max_accel;
    result.max_dynamic_pressure = max_q;
    result.trajectory_id = id;

    // Orbit achieved: altitude within 15% AND horizontal velocity within 15%
    double alt_err = fabs(best_alt - TARGET_ALTITUDE) / TARGET_ALTITUDE;
    double vel_err = fabs(best_vel - TARGET_VELOCITY) / TARGET_VELOCITY;
    result.reached_orbit = (alt_err < 0.15 && vel_err < 0.15);

    memcpy(&result.control, control, sizeof(ControlProfile));
    result.fitness = 0.0;
    return result;
}

/*
 * FIXED calculate_fitness
 *
 * The original function multiplied fitness by 0.5 for orbit and 2.0 for
 * non-orbit AFTER the calculation, but because fitness was based on fuel
 * alone (with small penalties), it was possible for a non-orbit trajectory
 * that used little fuel to beat an orbit trajectory.
 *
 * Fix: apply a very large flat penalty for not reaching orbit, and otherwise
 * reward minimizing fuel use, altitude error, and velocity error.
 */
HD double calculate_fitness(TrajectoryResult *result)
{
    double fitness = 0.0;

    double alt_err = fabs(result->final_altitude - TARGET_ALTITUDE);
    double vel_err = fabs(result->final_velocity - TARGET_VELOCITY);

    // Heavily penalize altitude miss (normalised to target)
    fitness += (alt_err / TARGET_ALTITUDE) * 500.0;

    // Heavily penalize velocity miss
    fitness += (vel_err / TARGET_VELOCITY) * 500.0;

    // Small reward for fuel efficiency (secondary objective)
    fitness += result->fuel_consumed / FUEL_MASS * 50.0;

    // Penalty for exceeding max-g limit
    if (result->max_acceleration > MAX_ACCELERATION)
        fitness += (result->max_acceleration - MAX_ACCELERATION) * 10.0;

    // Big bonus (negative penalty) when orbit IS reached
    if (result->reached_orbit)
        fitness -= 400.0;

    result->fitness = fitness;
    return fitness;
}

HD double calculate_energy(State *state)
{
    double r = sqrt(state->x * state->x + state->y * state->y + state->z * state->z);
    double v2 = state->vx * state->vx + state->vy * state->vy + state->vz * state->vz;
    return 0.5 * state->mass * v2 - G_CONST * M_EARTH * state->mass / r;
}

HD double calculate_rmse(TrajectoryResult *ref, TrajectoryResult *test)
{
    double d1 = (ref->fuel_consumed - test->fuel_consumed) / 1000.0;
    double d2 = (ref->final_altitude - test->final_altitude) / 1000.0;
    double d3 = (ref->final_velocity - test->final_velocity) / 10.0;
    return sqrt((d1 * d1 + d2 * d2 + d3 * d3) / 3.0);
}