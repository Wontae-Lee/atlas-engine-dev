#pragma once

#include <atlas/generator/generate.h>
#include <atlas/material/material_properties.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

namespace atlas {

struct SourceProbe {

    const Unit* units {};

    const Generate* generators {};

    const MaterialProperties* properties {};

    const std::size_t* shuffled_species {};

    Float3* positions {};

    Float3* velocities {};

    std::size_t* species {};

    int* active {};

    const Float3* flat_local_positions {};

    const int* flat_unit_indices {};

    float temperature {};

    int property_count {};

    std::uint64_t emission_seed {};
};

}
