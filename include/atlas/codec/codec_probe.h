#pragma once

/**
 * @file codec_probe.h
 * @brief Flat, device-copyable view over the per-cell universe state and
 *        allocation buffers a `Codec`'s `encode()`/`decode()` read and
 *        write. Same non-owning-probe pattern as `ColliderProbe`/
 *        `DsmcProbe` (see `collider_probe.h`).
 */

namespace atlas {

/**
 * @brief Non-owning snapshot of a `Codec`'s universe/fluid/searcher
 *        state and allocation buffers. See `Codec::make_probe` for how
 *        it is filled and `codec.h`/`knudsen_codec.h` for how each
 *        field is used.
 */
struct CodecProbe {
    /** Per-cell temperature, owned by the universe's
     *  `UniverseTemperatureState`; `nullptr` if that state doesn't
     *  exist. */
    const float* temperature_ptr {};
    /** Per-cell particle count, owned by
     *  `UniverseNumberParticleState`; `nullptr` if absent. */
    const float* number_particle_ptr {};
    /** Per-cell Knudsen number, owned by `UniverseKnudsenNumberState`;
     *  written by `KnudsenCodec::encode()`, read by `decode()`.
     *  `nullptr` if that state doesn't exist. */
    float* knudsen_number_ptr {};
    /** Per-cell solver index, owned by `Codec::allocated_solver()`;
     *  written by `decode()`. `nullptr` if the buffer is empty. */
    int* allocated_solver_ptr {};
    /** Per-cell solver-index override, owned by `Codec::fixed_solver()`.
     *  `nullptr` if the buffer is empty. */
    const int* fixed_solver_ptr {};
    /** Per-cell "exempt from automatic classification" flags, owned by
     *  `Codec::fixed_region()`. `nullptr` if the buffer is empty. */
    const int* fixed_region_ptr {};
    /** Searcher's particle indices sorted by cell. */
    const int* indices_ptr {};
    /** Per-cell start offset into `indices_ptr`. */
    const int* cell_start_ptr {};
    /** Per-cell (one-past-)end offset into `indices_ptr`. */
    const int* cell_end_ptr {};
    /** Total particle count in the fluid this codec observes. */
    int particle_count {};
    /** Number of spatial cells (length of every per-cell array above). */
    int cell_count {};
    /** Volume of one spatial cell (uniform-grid assumption). */
    float cell_volume {};
    /** Real molecules per simulated particle (used in number-density
     *  calculations, e.g. `KnudsenCodec::knudsen_number`). */
    float statistical_weight {};
};

}
