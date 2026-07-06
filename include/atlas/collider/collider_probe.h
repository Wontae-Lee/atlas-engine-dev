#pragma once

#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

namespace atlas {

struct ColliderProbe {

    const Unit* units {};

    const atlas::AABB* unit_bounds {};

    const SurfaceInteractionKernel* surface_interactions {};

    const std::uint8_t* flips {};

    atlas::AABB scene_bound {};

    Float3* positions {};

    Float3* velocities {};

    FluidInternalEnergy* internal_energies {};

    const std::size_t* species {};

    const MaterialProperties* materials {};

    int unit_count {};

    int interaction_count {};

    int flip_count {};

    int material_count {};

    int particle_count {};

    bool scene_bound_covers_units {};
};

}
