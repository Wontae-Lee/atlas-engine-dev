#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

namespace atlas {

// A view is the raw-pointer face of a set of universe states, gathered once on
// the host so a kernel can capture it by value. Views own nothing; a state the
// universe does not carry shows up as a null pointer.
//
// Pull one out with Universe::view<UniverseDsmcView>(). Adding a view means
// adding a struct here with a static make(Universe&) — nothing in Universe
// changes.

// The per-cell data a DSMC collision step reads and writes. number_particle is
// produced by the searcher; allocated_solver is optional (a null one means
// every cell belongs to this solver).
struct UniverseDsmcView final {

    const float* number_particle {};

    float* max_relative_speed {};

    float* max_sigma_g {};

    int* collision_count {};

    const int* allocated_solver {};

    int cell_count {};

    float cell_volume {};

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

    ATLAS_NODISCARD ATLAS_HOST bool
    is_complete() const noexcept {
        return number_particle != nullptr
            && max_relative_speed != nullptr
            && max_sigma_g != nullptr
            && collision_count != nullptr;
    }
};

}
