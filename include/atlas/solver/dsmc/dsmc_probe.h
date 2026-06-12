#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>

#include <cstddef>
#include <cstdint>

namespace atlas {

template <typename T>
struct DsmcProbe {
    Vector3<T>* velocity_ptr {};
    FluidInternalEnergy<T>* internal_energy_ptr {};
    const std::size_t* species_ptr {};
    const MaterialProperties<T>* properties_ptr {};
    T* number_particle_ptr {};
    T* max_relative_speed_ptr {};
    T* max_sigma_g_ptr {};
    T* collision_remainder_ptr {};
    int* collision_count_ptr {};
    const int* indices_ptr {};
    const int* cell_start_ptr {};
    const int* cell_end_ptr {};
    const T* universe_volume_ptr {};
    int particle_count {};
    int species_count {};
    int num_of_cells {};
    T cell_volume {};
    T statistical_weight {};
    DsmcKernel<T> kernel {};
    std::uint64_t collision_seed {};
};

}

namespace atlas {
}
