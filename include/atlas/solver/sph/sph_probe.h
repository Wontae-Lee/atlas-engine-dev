#pragma once

#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_kernel.h>

#include <cstddef>

namespace atlas {

struct SphProbe {

    const Float3* position_ptr {};

    Float3* velocity_ptr {};

    const std::size_t* species_ptr {};

    const MaterialProperties* properties_ptr {};

    float* number_particle_ptr {};

    Float3* field_force_ptr {};

    const int* indices_ptr {};

    const int* cell_start_ptr {};

    const int* cell_end_ptr {};

    const int* neighbor_offsets_ptr {};

    const int* neighbor_indices_ptr {};

    Float3 lower_corner {};

    Int3 grid_size {};

    float inverse_cell_size {};

    float cell_size {};

    int particle_count {};

    int cell_count {};

    int property_count {};

    SphKernel kernel {};
};

using SphSolverProbe = atlas::SphProbe;

}
