#pragma once

#include <atlas/material/material_properties.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

namespace atlas::fluid {

template <typename T>
struct SourceProbe {
    const Unit<T>* units {};
    const GenerateOperator<T>* generators {};
    const MaterialProperties<T>* properties {};
    const std::size_t* shuffled_species {};

    Vector3<T>* positions {};
    Vector3<T>* velocities {};
    std::size_t* species {};
    int* active {};

    // Flat contiguous array of all local emission positions across every unit.
    const Vector3<T>* flat_local_positions {};

    // Unit index for each entry in flat_local_positions.
    const int* flat_unit_indices {};

    T temperature {};
    int property_count {};
    std::uint64_t emission_seed {};
};

}

namespace atlas {

template <typename T>
using SourceProbe = atlas::fluid::SourceProbe<T>;

}
