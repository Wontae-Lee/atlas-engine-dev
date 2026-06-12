#pragma once

#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_kernel.h>

#include <cstddef>

namespace atlas {

template <typename T>
struct SphProbe {
    const Vector3<T>* position_ptr {};
    Vector3<T>* velocity_ptr {};
    const std::size_t* species_ptr {};
    const MaterialProperties<T>* properties_ptr {};
    T* number_particle_ptr {};
    Vector3<T>* field_force_ptr {};
    const int* indices_ptr {};
    const int* cell_start_ptr {};
    const int* cell_end_ptr {};
    const int* neighbor_offsets_ptr {};
    const int* neighbor_indices_ptr {};
    Vector3<T> lower_corner {};
    Vector3<int> grid_size {};
    T inverse_cell_size {};
    T cell_size {};
    int particle_count {};
    int num_of_cells {};
    int num_of_properties {};
    SphKernel<T> kernel {};
};

template <typename T>
using SphSolverProbe = SphProbe<T>;

}

namespace atlas {
template <typename T>
using SphSolverProbe = atlas::SphProbe<T>;

}
