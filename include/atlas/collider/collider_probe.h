#pragma once

#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/unit/unit.h>

#include <cstdint>

namespace atlas::system {

template <typename T>
struct ColliderProbe {
    const Unit<T>* units {};
    const ColliderSurfaceInteraction<T>* surface_interactions {};
    const std::uint8_t* flips {};

    Vector3<T>* positions {};
    Vector3<T>* velocities {};

    int unit_count {};
    int interaction_count {};
    int flip_count {};
    int particle_count {};
};

}

namespace atlas {

template <typename T>
using ColliderProbe = atlas::system::ColliderProbe<T>;

}
