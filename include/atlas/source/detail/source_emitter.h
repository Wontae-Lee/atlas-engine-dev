#pragma once

#include <atlas/core/macros.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/shuffle/shuffle.h>
#include <atlas/source/source_probe.h>

#include <cstddef>

/**
 * @file source_emitter.h
 * @brief Writes `emit_count` new particles into the fluid's tail
 *        buffer slots, using `detail::SourceCacheBuilder`'s cached
 *        candidate positions and the per-candidate shuffled species
 *        assignment.
 *
 * @details
 * Per new particle: transforms its cached local-frame candidate
 * position into world space via the owning unit's `sync()`,
 * draws a velocity from the assigned species' `Generate` at the
 * source's configured temperature (see `generate.h` for the
 * available velocity-sampling models — Maxwellian/uniform/jittering),
 * and marks it active. `Shuffle`/`emission_seed` derive a
 * per-particle hashed sample index rather than a stateful RNG stream —
 * the same stateless-hash-RNG pattern used throughout Atlas's device
 * code (see `maxwellian_surface_interaction.h`).
 */

namespace atlas::detail {

/** @brief Device-side emission kernel launcher; see this file's
 *  top-of-file documentation for what each new particle is assigned. */
class SourceEmitter final {
public:
    /**
     * @brief Writes `emit_count` particles starting at fluid index
     *        `dst_offset`, sourced from `probe`'s cached candidate
     *        positions/shuffled species `[0, emit_count)`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(const SourceProbe& probe,
         const std::size_t dst_offset,
         const std::size_t emit_count) const {
        const Shuffle shuffle {};
        const auto device_probe = probe;

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            static_cast<int>(emit_count),
            [=] ATLAS_ALL_DEVICE(const int i) {
                const int unit_index  = device_probe.flat_unit_indices[i];
                const int dst         = static_cast<int>(dst_offset) + i;
                const std::size_t sid = device_probe.shuffled_species[i];

                if (sid >= static_cast<std::size_t>(device_probe.property_count)) {
                    return;
                }

                const auto sample_seed = static_cast<unsigned int>(shuffle(dst, device_probe.emission_seed));

                Vector3 world_pos;
                device_probe.units[unit_index].sync().sync_to_world(
                    device_probe.flat_local_positions[i],
                    world_pos);

                device_probe.positions[dst]  = world_pos;
                device_probe.velocities[dst] = device_probe.generators[sid].generate(
                    sample_seed,
                    device_probe.temperature,
                    device_probe.properties[sid].molecular_mass);
                device_probe.species[dst] = sid;
                device_probe.active[dst]  = 1;
            });
    }
};

}
