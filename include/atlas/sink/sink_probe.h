#pragma once

#include <atlas/sink/despawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstddef>

namespace atlas::fluid {

template <typename T>
struct SinkProbe {
    const Unit<T>* units {};
    const DespawnOperator<T>* despawn_operators {};
    const Vector3<T>* positions {};
    const Vector3<T>* velocities {};
    int* active {};

    int unit_count {};
    int despawn_operator_count {};
    std::size_t particle_count {};

    bool flip {};
    T tolerance {};
    T time_step {};
};

}

namespace atlas {

template <typename T>
using SinkProbe = atlas::fluid::SinkProbe<T>;

}
