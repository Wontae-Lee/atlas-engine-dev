#pragma once

namespace atlas {

struct CodecProbe {

    const float* temperature_ptr {};

    const float* number_particle_ptr {};

    float* knudsen_number_ptr {};

    int* allocated_solver_ptr {};

    const int* fixed_solver_ptr {};

    const int* fixed_region_ptr {};

    const int* indices_ptr {};

    const int* cell_start_ptr {};

    const int* cell_end_ptr {};

    int particle_count {};

    int cell_count {};

    float cell_volume {};

    float statistical_weight {};
};

}
