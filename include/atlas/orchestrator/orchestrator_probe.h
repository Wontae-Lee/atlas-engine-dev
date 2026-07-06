#pragma once

#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>

#include <cstddef>

namespace atlas {

struct OrchestratorProbe {
    Float3* velocity_ptr {};
    const std::size_t* species_ptr {};
    const MaterialProperties* properties_ptr {};
    const Float3* field_force_ptr {};
    const Float3* gravity_ptr {};
    const int* indices_ptr {};
    const int* cell_start_ptr {};
    const int* cell_end_ptr {};
    int particle_count {};
    int cell_count {};
    int species_count {};
    int field_force_cell_count {};
    int gravity_cell_count {};
};

}
