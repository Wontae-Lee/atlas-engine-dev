#pragma once

#include <atlas/math/math.h>

/**
 * @file measurer_probe.h
 * @brief Flat, device-copyable view over the per-cell field states and
 *        per-particle velocity data a `Measurer` reads/writes. Same
 *        non-owning-probe pattern as `ColliderProbe` (see
 *        `collider_probe.h`).
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of a `Measurer`'s universe/fluid/searcher
 *        state, passed by value into measurement device kernels. See
 *        `Measurer::make_probe` for how it is filled and
 *        `boltzmann_measurer.h` for how each field is used.
 */
struct MeasurerProbe {
    /** Per-cell derived temperature, owned by `UniverseTemperatureState`. */
    float* field_temperature_ptr {};
    /** Per-cell mean particle velocity, owned by
     *  `UniverseBulkVelocityState`. */
    Vector3* bulk_velocity_ptr {};
    /** Per-cell summed squared velocity fluctuation
     *  (`sum |v - bulk_velocity|^2`), owned by
     *  `UniverseThermalEnergyState`. */
    float* thermal_energy_ptr {};
    /** Per-cell particle count, owned by `UniverseNumberParticleState`. */
    float* number_particle_ptr {};
    /** Particle world-space velocities, owned by `Fluid` (read-only
     *  input). */
    const Vector3* velocity_ptr {};
    /** Per-particle temperature, owned by `FluidTemperatureState`;
     *  `nullptr` if the fluid does not track it. */
    float* particle_temperature_ptr {};
    /** Searcher's particle indices sorted by cell. */
    const int* indices_ptr {};
    /** Per-cell start offset into `indices_ptr`. */
    const int* cell_start_ptr {};
    /** Per-cell (one-past-)end offset into `indices_ptr`. */
    const int* cell_end_ptr {};
    /** Length of `velocity_ptr`/`particle_temperature_ptr`. */
    int particle_count {};
    /** Number of spatial cells (length of every per-cell array above). */
    int cell_count {};
};

}
