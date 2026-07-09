#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cstddef>

namespace atlas {

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