#pragma once

#include <atlas/sink/despawn.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>

namespace atlas {

struct SinkProbe {

    const Unit* units {};

    const atlas::AABB* unit_bounds {};

    const Despawn* despawn_operators {};

    const Float3* positions {};

    const Float3* velocities {};

    int* active {};

    int unit_count {};

    int despawn_operator_count {};

    std::size_t particle_count {};

    bool flip {};

    float tolerance {};

    float time_step {};
};

}
