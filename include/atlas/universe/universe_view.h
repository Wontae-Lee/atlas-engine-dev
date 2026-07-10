#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

namespace atlas {

/**
 * @brief Device-capturable snapshot of the Universe fields the DSMC solver needs.
 *
 * A trivially-copyable bundle of raw device pointers plus the grid scalars,
 * gathered once on the host (`make`) so a `__host__ __device__` kernel lambda
 * can capture it by value — a `Universe` and its state store cannot cross onto
 * the device. Each pointer is borrowed: it aliases the buffer inside the
 * corresponding `UniverseState` and stays valid only until that state is
 * removed or reallocated, so the view must be re-gathered afterwards. A field
 * whose state is absent stays null; `is_complete()` reports whether the
 * mandatory ones are present.
 *
 * @note The pointer const-ness mirrors the solver's access: inputs it only
 *       reads (`number_particle`, `allocated_solver`) are `const`, while the
 *       counters it writes back are mutable.
 */
struct UniverseDsmcView final {

    const float* number_particle {}; ///< Read-only per-cell real-particle count `N`.

    float* max_relative_speed {}; ///< Per-cell max pair relative speed (written each step).

    float* max_sigma_g {}; ///< Per-cell NTC bound (sigma*g)_max (read then raised).

    int* collision_count {}; ///< Per-cell candidate-collision count / scheduling slot.

    const int* allocated_solver {}; ///< Read-only per-cell owning-solver index; null = all own.

    int cell_count {}; ///< Total number of grid cells.

    float cell_volume {}; ///< Volume of one cell, used as the collision-rate denominator.

    /**
     * @brief Gather the view from a live Universe.
     *
     * Fills each pointer from the matching `UniverseState` if present, leaving
     * it null otherwise, and copies `cell_count` and `cell_volume`. Every
     * pointer is a raw device pointer obtained via `raw_pointer_cast`.
     *
     * @param universe Universe to snapshot; its state buffers must outlive the view.
     * @return The populated view, by value.
     */
    ATLAS_NODISCARD ATLAS_HOST static UniverseDsmcView
    make(Universe& universe) {
        UniverseDsmcView view {};

        if (auto* state = universe.state<UniverseNumberParticleState>(); state != nullptr) {
            view.number_particle = atlas::raw_pointer_cast(state->data().data());
        }

        if (auto* state = universe.state<UniverseMaxRelativeSpeedState>(); state != nullptr) {
            view.max_relative_speed = atlas::raw_pointer_cast(state->data().data());
        }

        if (auto* state = universe.state<UniverseMaxSigmaGState>(); state != nullptr) {
            view.max_sigma_g = atlas::raw_pointer_cast(state->data().data());
        }

        if (auto* state = universe.state<UniverseCollisionCountState>(); state != nullptr) {
            view.collision_count = atlas::raw_pointer_cast(state->data().data());
        }

        if (auto* state = universe.state<UniverseAllocatedSolverState>(); state != nullptr) {
            view.allocated_solver = atlas::raw_pointer_cast(state->data().data());
        }

        view.cell_count  = universe.cell_count();
        view.cell_volume = universe.cell_volume();

        return view;
    }

    /**
     * @brief Whether every field the DSMC solver requires was gathered.
     *
     * `allocated_solver` is intentionally excluded: a null owning-solver buffer
     * is the valid "all solvers own every cell" case, not a missing field.
     *
     * @return `true` if `number_particle`, `max_relative_speed`, `max_sigma_g`
     *         and `collision_count` are all non-null.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    is_complete() const noexcept {
        return number_particle != nullptr
            && max_relative_speed != nullptr
            && max_sigma_g != nullptr
            && collision_count != nullptr;
    }
};

}