#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas::system {

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

    T* const temperature          = particle.temperature;
    const Vector3<T>* position    = particle.pos;
    const int* const indices      = searcher.indices;
    const int* const cell_start   = searcher.cell_start;
    const int* const cell_end     = searcher.cell_end;
    const Vector3<int> grid_size  = domain.grid_size;
    const Vector3<T> lower_corner = domain.lower_corner;
    const T inv_h                 = domain.inv_h;
    const int num_of_cells        = domain.num_of_cells;
    const int particle_count      = particle.particle_count;

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
typename AverageThermometer<T>::Builder
AverageThermometer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
AverageThermometer<T>::AverageThermometer(const atlas::system::ThermometerOperator<T>& thermometer_operator,
                                          const MeasureModeType measure_mode) noexcept
    : Thermometer<T>(measure_mode)
    , _thermometer_operator(thermometer_operator) {
}

template <typename T>
AverageThermometer<T>::AverageThermometer(
    const atlas::system::AverageThermometerOperator<T>& thermometer_operator,
    const MeasureModeType measure_mode) noexcept
    : Thermometer<T>(measure_mode)
    , _thermometer_operator(thermometer_operator) {
}

template <typename T>
void
AverageThermometer<T>::measure(DomainDeviceProbe<T> domain,
                               SpatialHashingProbe<T> searcher,
                               FluidDeviceProbe<T> particle) {
    if (domain.type == DomainType::isothermal) {
        atlas::logger::error()
            << "AverageThermometer: measure() is forbidden when the domain type is isothermal.";
        throw std::runtime_error("AverageThermometer: measure() is forbidden when the domain type is isothermal.");
    }

    _thermometer_operator.measure(domain, searcher, particle, this->measure_mode());
}

template <typename T>
bool
AverageThermometer<T>::is_valid() const noexcept {
    return _thermometer_operator.type == ThermometerType::Average;
}

template <typename T>
ThermometerType
AverageThermometer<T>::type() const noexcept {
    return ThermometerType::Average;
}

template <typename T>
void
AverageThermometer<T>::set_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
}

template <typename T>
const atlas::system::ThermometerOperator<T>&
AverageThermometer<T>::thermometer_operator() const noexcept {
    return _thermometer_operator;
}

template <typename T>
typename AverageThermometer<T>::Builder&
AverageThermometer<T>::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {
    _measure_mode = measure_mode;
    return *this;
}

template <typename T>
typename AverageThermometer<T>::Builder&
AverageThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::ThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = thermometer_operator;
    return *this;
}

template <typename T>
typename AverageThermometer<T>::Builder&
AverageThermometer<T>::Builder::with_thermometer_operator(
    const atlas::system::AverageThermometerOperator<T>& thermometer_operator) noexcept {
    _thermometer_operator = atlas::system::ThermometerOperator<T>(thermometer_operator);
    return *this;
}

template <typename T>
AverageThermometer<T>
AverageThermometer<T>::Builder::build() const {
    return AverageThermometer<T>(_thermometer_operator, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<AverageThermometer<T>>
AverageThermometer<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<AverageThermometer<T>>(build());
}

}
