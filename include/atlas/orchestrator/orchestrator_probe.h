#pragma once

#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>

#include <cstddef>

namespace atlas {

template <typename T>
struct OrchestratorProbe {
    Vector3<T>* velocity_ptr {};
    const std::size_t* species_ptr {};
    const MaterialProperties<T>* properties_ptr {};
    const Vector3<T>* field_force_ptr {};
    const Vector3<T>* gravity_ptr {};
    const int* indices_ptr {};
    const int* cell_start_ptr {};
    const int* cell_end_ptr {};
    int particle_count {};
    int num_of_cells {};
    int num_of_species {};
    int field_force_cell_count {};
    int gravity_cell_count {};
};

}