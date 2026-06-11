#pragma once

#include <atlas/math/math.h>

namespace atlas::system {

template <typename T>
struct CodecProbe {
    const T* temperature_ptr {};
    const T* number_particle_ptr {};
    T* knudsen_number_ptr {};
    int* allocated_solver_ptr {};
    const int* fixed_solver_ptr {};
    const int* fixed_region_ptr {};
    const int* indices_ptr {};
    const int* cell_start_ptr {};
    const int* cell_end_ptr {};
    int particle_count {};
    int num_of_cells {};
    T cell_volume {};
    T statistical_weight {};
};

}

namespace atlas {

template <typename T>
using CodecProbe = atlas::system::CodecProbe<T>;

}
