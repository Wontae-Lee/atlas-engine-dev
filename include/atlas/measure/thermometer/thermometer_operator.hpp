#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <cmath>

namespace atlas::system {

template <typename T>
void
VarianceThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                        const SpatialHashingProbe<T>& searcher,
                                        const FluidDeviceProbe<T>& particle,
                                        const MeasureModeType measure_mode) const {
    DeviceBuffer<T> scratch_field {};
    T* field = nullptr;

    if (measure_mode == MeasureModeType::Field || measure_mode == MeasureModeType::All) {
        field = domain.field_temperature;
    } else {
        scratch_field.resize(static_cast<std::size_t>(domain.num_of_cells));
        field = atlas::raw_pointer_cast(scratch_field.data());
    }

    const Vector3<T>* const position                    = particle.pos;
    const Vector3<T>* const velocity                    = particle.vel;
    const MatrialProperties<T>* const particle_property = particle.particle_property;
    T* const particle_temperature                       = particle.temperature;
    const size_t* const species                         = particle.species;
    const int* const indices                            = searcher.indices;
    const int* const cell_start                         = searcher.cell_start;
    const int* const cell_end                           = searcher.cell_end;
    const Vector3<int> grid_size                        = domain.grid_size;
    const Vector3<T> lower_corner                       = domain.lower_corner;
    const T inv_h                                       = domain.inv_h;
    const int num_of_cells                              = domain.num_of_cells;
    const int particle_count                            = particle.particle_count;

    const bool update_fluid = measure_mode == MeasureModeType::Fluid || measure_mode == MeasureModeType::All;

    if (field == nullptr || position == nullptr || velocity == nullptr || particle_property == nullptr
        || species == nullptr || indices == nullptr || cell_start == nullptr || cell_end == nullptr
        || num_of_cells <= 0 || particle_count < 0 || (update_fluid && particle_temperature == nullptr)) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start[cell];
            if (begin < 0) {
                field[cell] = T(0);
                return;
            }

            const int end = cell_end[cell];
            Vector3<T> mean_velocity { T(0), T(0), T(0) };
            T momentum_weight_sum = T(0);
            int count             = 0;

            for (int k = begin; k < end; ++k) {
                const int particle_index = indices[k];
                if (particle_index < 0 || particle_index >= particle_count) continue;
                const size_t species_index = species[particle_index];
                const T mass               = particle_property[species_index].mass;
                mean_velocity += velocity[particle_index] * mass;
                momentum_weight_sum += mass;
                ++count;
            }

            if (count <= 0 || !(momentum_weight_sum > T(0))) {
                field[cell] = T(0);
                return;
            }

            mean_velocity /= momentum_weight_sum;

            T thermal_energy_sum = T(0);
            for (int k = begin; k < end; ++k) {
                const int particle_index = indices[k];
                if (particle_index < 0 || particle_index >= particle_count) continue;

                const size_t species_index = species[particle_index];
                const T mass               = particle_property[species_index].mass;
                const Vector3<T> dv        = velocity[particle_index] - mean_velocity;
                thermal_energy_sum += mass * (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            }

            field[cell] = thermal_energy_sum
                / (static_cast<T>(3) * static_cast<T>(atlas::boltzmann_constant) * static_cast<T>(count));
        });

    if (update_fluid) {
        const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };
        const Vector3<int> lo { 0, 0, 0 };

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            particle_count,
            [=] ATLAS_DEVICE(const int i) {
                const Vector3<T> rel = (position[i] - lower_corner) * inv_h;
                Vector3<int> ijk     = atlas::math::floor(rel).template cast_to<int>();
                ijk                  = atlas::math::clamp(ijk, lo, hi);

                const int cell          = ijk.x + ijk.y * grid_size.x + ijk.z * grid_size.x * grid_size.y;
                particle_temperature[i] = field[cell];
            });
    }
}

template <typename T>
void
AverageThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                       const SpatialHashingProbe<T>& searcher,
                                       const FluidDeviceProbe<T>& particle,
                                       const MeasureModeType measure_mode) const {
    DeviceBuffer<T> scratch_field {};
    T* field = nullptr;

    if (measure_mode == MeasureModeType::Field || measure_mode == MeasureModeType::All) {
        field = domain.field_temperature;
    } else {
        scratch_field.resize(static_cast<std::size_t>(domain.num_of_cells));
        field = atlas::raw_pointer_cast(scratch_field.data());
    }

    T* const temperature        = particle.temperature;
    const Vector3<T>* position  = particle.pos;
    const int* const indices    = searcher.indices;
    const int* const cell_start = searcher.cell_start;
    const int* const cell_end   = searcher.cell_end;
    const Vector3<int> grid_size = domain.grid_size;
    const Vector3<T> lower_corner = domain.lower_corner;
    const T inv_h               = domain.inv_h;
    const int num_of_cells      = domain.num_of_cells;
    const int particle_count    = particle.particle_count;

    const bool update_fluid = measure_mode == MeasureModeType::Fluid || measure_mode == MeasureModeType::All;

    if (field == nullptr || temperature == nullptr || indices == nullptr || cell_start == nullptr || cell_end == nullptr
        || num_of_cells <= 0 || particle_count < 0 || (update_fluid && position == nullptr)) {
        return;
    }

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start[cell];
            if (begin < 0) {
                field[cell] = T(0);
                return;
            }

            const int end = cell_end[cell];
            T sum         = T(0);
            int count     = 0;

            for (int k = begin; k < end; ++k) {
                const int particle_index = indices[k];
                if (particle_index < 0 || particle_index >= particle_count) continue;
                sum += temperature[particle_index];
                ++count;
            }

            field[cell] = count > 0 ? sum / static_cast<T>(count) : T(0);
        });

    if (update_fluid) {
        const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };
        const Vector3<int> lo { 0, 0, 0 };

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            particle_count,
            [=] ATLAS_DEVICE(const int i) {
                const Vector3<T> rel = (position[i] - lower_corner) * inv_h;
                Vector3<int> ijk     = atlas::math::floor(rel).template cast_to<int>();
                ijk                  = atlas::math::clamp(ijk, lo, hi);

                const int cell = ijk.x + ijk.y * grid_size.x + ijk.z * grid_size.x * grid_size.y;
                temperature[i] = field[cell];
            });
    }
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator() noexcept
    : type(ThermometerType::Average) {
    new (&average) AverageThermometerOperator<T> {};
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerType type_) noexcept
    : type(type_) {
    switch (type) {
    case ThermometerType::Variance:
        new (&variance) VarianceThermometerOperator<T> {};
        return;
    case ThermometerType::Average:
        new (&average) AverageThermometerOperator<T> {};
        return;
    default:
        type = ThermometerType::Average;
        new (&average) AverageThermometerOperator<T> {};
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
ThermometerOperator<T>::ThermometerOperator(const AverageThermometerOperator<T>& op) noexcept
    : type(ThermometerType::Average) {
    new (&average) AverageThermometerOperator<T>(op);
}

template <typename T>
void
ThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                const SpatialHashingProbe<T>& searcher,
                                const FluidDeviceProbe<T>& particle,
                                const MeasureModeType measure_mode) const {
    switch (type) {
    case ThermometerType::Variance:
        variance.measure(domain, searcher, particle, measure_mode);
        return;
    case ThermometerType::Average:
        average.measure(domain, searcher, particle, measure_mode);
        return;
    }

    average.measure(domain, searcher, particle, measure_mode);
}

template <typename T>
void
ThermometerOperator<T>::destroy_active() noexcept {
    switch (type) {
    case ThermometerType::Variance:
        variance.~VarianceThermometerOperator<T>();
        return;
    case ThermometerType::Average:
        average.~AverageThermometerOperator<T>();
        return;
    }

    average.~AverageThermometerOperator<T>();
}

template <typename T>
void
ThermometerOperator<T>::copy_from(const ThermometerOperator& other) noexcept {
    switch (type) {
    case ThermometerType::Variance:
        new (&variance) VarianceThermometerOperator<T>(other.variance);
        return;
    case ThermometerType::Average:
        new (&average) AverageThermometerOperator<T>(other.average);
        return;
    default:
        type = ThermometerType::Average;
        new (&average) AverageThermometerOperator<T>(other.average);
        return;
    }
}

}
