#pragma once

#include <atlas/math/math.h>

namespace atlas::system {

template <typename T>
struct MeasurerProbe {
    T* field_temperature_ptr {};
    Vector3<T>* bulk_velocity_ptr {};
    T* thermal_energy_ptr {};
    T* number_particle_ptr {};
    const Vector3<T>* velocity_ptr {};
    T* particle_temperature_ptr {};
    const int* indices_ptr {};
    const int* cell_start_ptr {};
    const int* cell_end_ptr {};
    int particle_count {};
    int num_of_cells {};
};

}

namespace atlas {

template <typename T>
using MeasurerProbe = atlas::system::MeasurerProbe<T>;

}
