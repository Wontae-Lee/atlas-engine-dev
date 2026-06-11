#pragma once

#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

namespace atlas::system {

template <typename T>
struct ColliderProbe {
    const Unit<T>* units {};
    const atlas::spatial::AxisAlignedBoundingBox<T>* unit_bounds {};
    const SurfaceInteractionKernel<T>* surface_interactions {};
    const std::uint8_t* flips {};
    atlas::spatial::AxisAlignedBoundingBox<T> scene_bound {};

    Vector3<T>* positions {};
    Vector3<T>* velocities {};
    fluid::FluidInternalEnergy<T>* internal_energies {};
    const std::size_t* species {};
    const MaterialProperties<T>* materials {};

    int unit_count {};
    int interaction_count {};
    int flip_count {};
    int material_count {};
    int particle_count {};
    bool scene_bound_covers_units {};
};

}

namespace atlas {

template <typename T>
using ColliderProbe = atlas::system::ColliderProbe<T>;

}
