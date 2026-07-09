#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cstddef>

namespace atlas {

// A view is the raw-pointer face of a set of fluid states, gathered once on the
// host so a kernel can capture it by value. Views own nothing; a state the
// fluid does not carry shows up as a null pointer.
//
// Pull one out with Fluid::view<FluidDsmcView>(). Adding a view means adding a
// struct here with a static make(Fluid&) — nothing in Fluid changes.

// The per-particle data a DSMC collision step reads and writes: it scatters
// velocities in place and looks each particle's species up in its own table.
struct FluidDsmcView final {

    Float3* velocity {};

    const std::size_t* species {};

    int particle_count {};

    float statistical_weight {};

    ATLAS_NODISCARD ATLAS_HOST static FluidDsmcView
    make(Fluid& fluid) {
        FluidDsmcView view {};

        if (auto* state = fluid.state<FluidVelocityState>(); state != nullptr) {
            view.velocity = atlas::raw_pointer_cast(state->data().data());
        }

        if (auto* state = fluid.state<FluidSpeciesState>(); state != nullptr) {
            view.species = atlas::raw_pointer_cast(state->data().data());
        }

        view.particle_count     = static_cast<int>(fluid.particle_count());
        view.statistical_weight = fluid.statistical_weight();

        return view;
    }

    ATLAS_NODISCARD ATLAS_HOST bool
    is_complete() const noexcept {
        return velocity != nullptr && species != nullptr;
    }
};

}
