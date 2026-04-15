#pragma once

namespace atlas::system {

template <typename T>
void
VarianceThermometerOperator<T>::measure(const Universe<T>& domain,
                                        const SpatialHashingProbe<T>& searcher,
                                        const FluidDeviceProbe<T>& particle,
                                        const MeasureModeType measure_mode) const {

    const auto num_of_cells = domain.num_of_cells;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = searcher.cell_start[cell];

            const int end = searcher.cell_end[cell];

            Vector3<T> mean_velocity { T(0), T(0), T(0) };

            T momentum_weight_sum = T(0);

            int count = 0;

            for (int k = begin; k < end; ++k) {

                const int particle_index = searcher.indices[k];

                if (particle_index < 0 || particle_index >= particle.particle_count) continue;

                const size_t species_index = particle.species[particle_index];

                const T mass = particle.particle_property[species_index].mass;

                mean_velocity += particle.vel[particle_index] * mass;

                momentum_weight_sum += mass;

                ++count;
            }

            if (count <= 0 || !(momentum_weight_sum > T(0))) {
                domain.field_temperature[cell] = T(0);
                return;
            }

            mean_velocity /= momentum_weight_sum;

            T thermal_energy_sum = T(0);

            for (int k = begin; k < end; ++k) {

                const int particle_index = searcher.indices[k];

                if (particle_index < 0 || particle_index >= particle.particle_count) continue;

                const size_t species_index = particle.species[particle_index];

                const T mass = particle.particle_property[species_index].mass;

                const Vector3<T> dv = particle.vel[particle_index] - mean_velocity;

                thermal_energy_sum += mass * (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            }

            domain.field_temperature[cell] = thermal_energy_sum
                / (static_cast<T>(3)
                   * static_cast<T>(atlas::boltzmann_constant)
                   * static_cast<T>(count));
        });

    if (measure_mode == MeasureModeType::Fluid || measure_mode == MeasureModeType::All) {

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                const int begin = searcher.cell_start[cell];

                const int end = searcher.cell_end[cell];

                for (int k = begin; k < end; ++k) {

                    const int particle_index             = searcher.indices[k];
                    particle.temperature[particle_index] = domain.field_temperature[cell];
                }
            });
    }
}
template <typename T>
ThermometerOperator<T>::ThermometerOperator() noexcept
    : type(ThermometerType::Variance) {

    new (&variance) VarianceThermometerOperator<T> {};
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerType type_) noexcept
    : type(type_) {

    switch (type) {
    case ThermometerType::Variance:

        new (&variance) VarianceThermometerOperator<T> {};
        return;

    default:

        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerOperator& other) noexcept
    : type(other.type) {

    copy_from(other);
}

template <typename T>
ThermometerOperator<T>&
ThermometerOperator<T>::operator=(const ThermometerOperator& other) noexcept {

    if (this == &other) return *this;

    destroy_active();

    type = other.type;

    copy_from(other);

    return *this;
}

template <typename T>
ThermometerOperator<T>::~ThermometerOperator() noexcept {

    destroy_active();
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const VarianceThermometerOperator<T>& op) noexcept
    : type(ThermometerType::Variance) {

    new (&variance) VarianceThermometerOperator<T>(op);
}

template <typename T>
void
ThermometerOperator<T>::measure(const Universe<T>& domain,
                                const SpatialHashingProbe<T>& searcher,
                                const FluidDeviceProbe<T>& particle,
                                const MeasureModeType measure_mode) const {

    switch (type) {
    case ThermometerType::Variance:

        variance.measure(domain, searcher, particle, measure_mode);
        return;

    default:

        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
void
ThermometerOperator<T>::destroy_active() noexcept {

    switch (type) {
    case ThermometerType::Variance:

        variance.~VarianceThermometerOperator<T>();
        return;

    default:

        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
void
ThermometerOperator<T>::copy_from(const ThermometerOperator& other) noexcept {

    switch (type) {
    case ThermometerType::Variance:

        new (&variance) VarianceThermometerOperator<T>(other.variance);
        return;

    default:

        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

}