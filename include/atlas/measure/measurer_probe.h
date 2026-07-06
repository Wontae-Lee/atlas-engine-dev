#pragma once

#include <atlas/math/math.h>

namespace atlas {

struct MeasurerProbe {

    float* field_temperature_ptr {};

    Float3* bulk_velocity_ptr {};

    float* thermal_energy_ptr {};

    float* number_particle_ptr {};

    const Float3* velocity_ptr {};

    float* particle_temperature_ptr {};

    const int* indices_ptr {};

    const int* cell_start_ptr {};

    const int* cell_end_ptr {};

    int particle_count {};

    int cell_count {};
};

}
