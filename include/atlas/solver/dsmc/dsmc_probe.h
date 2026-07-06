#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>

#include <cstddef>
#include <cstdint>

namespace atlas {

struct DsmcProbe {

    Float3* velocity_ptr {};

    FluidInternalEnergy* internal_energy_ptr {};

    const std::size_t* species_ptr {};

    const MaterialProperties* properties_ptr {};

    float* number_particle_ptr {};

    float* max_relative_speed_ptr {};

    float* max_sigma_g_ptr {};

    float* collision_remainder_ptr {};

    int* collision_count_ptr {};

    const int* indices_ptr {};

    const int* cell_start_ptr {};

    const int* cell_end_ptr {};

    const float* universe_volume_ptr {};

    int particle_count {};

    int species_count {};

    int cell_count {};

    float cell_volume {};

    float statistical_weight {};

    DsmcKernel kernel {};

    std::uint64_t collision_seed {};
};

}
