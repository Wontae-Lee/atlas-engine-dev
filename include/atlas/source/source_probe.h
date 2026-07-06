#pragma once

#include <atlas/generator/generate.h>
#include <atlas/material/material_properties.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

/**
 * @file source_probe.h
 * @brief Flat, device-copyable view over everything one particle-
 *        emission pass needs: the cached candidate spawn lattice, each
 *        species' velocity generator, and the fluid's particle buffers.
 *        Same non-owning-probe pattern as `ColliderProbe`
 *        (see `collider_probe.h`).
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of a `Source` + its `Fluid`, passed by
 *        value into `Source`'s device kernel. See
 *        `Source::make_probe` for how it is filled and
 *        `source.h` for how each field is
 *        used.
 */
struct SourceProbe {
    /** Source units (geometry + sync/transform), owned by `Source`. */
    const Unit* units {};
    /** Per-species velocity samplers, owned by `Fluid::generators()`
     *  (see `Generate`). */
    const Generate* generators {};
    /** Per-species material properties (molecular mass, ...). */
    const MaterialProperties* properties {};
    /** Per-cached-candidate species assignment, already randomized by
     *  `Source`; only the first `emit_count`
     *  entries are consumed by one `emit()` call. */
    const std::size_t* shuffled_species {};

    /** Particle world-space positions, owned by `Fluid`; written for
     *  newly emitted particles at `[dst_offset, dst_offset+emit_count)`. */
    Float3* positions {};
    /** Particle world-space velocities, owned by `Fluid`; written for
     *  newly emitted particles, drawn from the assigned species'
     *  generator at `temperature`. */
    Float3* velocities {};
    /** Particle species indices, owned by `Fluid`; written for newly
     *  emitted particles. */
    std::size_t* species {};
    /** Particle active flags, owned by `Fluid`; set to `1` for newly
     *  emitted particles. */
    int* active {};

    /** Cached candidate spawn positions in each candidate's owning
     *  unit's local frame (see `Source`). */
    const Float3* flat_local_positions {};

    /** Owning unit index for each entry of `flat_local_positions`
     *  (parallel array). */
    const int* flat_unit_indices {};

    /** Emission temperature passed to each species' velocity generator. */
    float temperature {};
    /** Length of `generators`/`properties`. */
    int property_count {};
    /** Per-emission random seed mixed into each particle's velocity
     *  sample (the same stateless-hash-RNG pattern used throughout
     *  Atlas's device code). */
    std::uint64_t emission_seed {};
};

}
