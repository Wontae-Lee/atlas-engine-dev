#pragma once
#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <stdexcept>
namespace atlas::system {
template <typename T>
typename BoltzmanMeasurer<T>::Builder
BoltzmanMeasurer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
BoltzmanMeasurer<T>::BoltzmanMeasurer(UniverseHostPtr<T> universe,
                                      FluidHostPtr<T> fluid,
                                      SpatialHashingSearcherHostPtr<T> searcher,
                                      const MeasureModeType measure_mode) noexcept
    : Measurer<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _measure_mode(measure_mode) {
    if (this->_universe != nullptr) {
        const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());
        if (!this->_universe->template has_state<atlas::universe::UniverseTemperatureState<T>>()) {
            this->_universe->template emplace_state<atlas::universe::UniverseTemperatureState<T>>(number_of_cells);
        }
        if (!this->_universe->template has_state<atlas::universe::UniverseBulkVelocityState<T>>()) {
            this->_universe->template emplace_state<atlas::universe::UniverseBulkVelocityState<T>>(number_of_cells);
        }
        if (!this->_universe->template has_state<atlas::universe::UniverseThermalEnergyState<T>>()) {
            this->_universe->template emplace_state<atlas::universe::UniverseThermalEnergyState<T>>(number_of_cells);
        }
        if (!this->_universe->template has_state<atlas::universe::UniverseNumberParticleState<T>>()) {
            this->_universe->template emplace_state<atlas::universe::UniverseNumberParticleState<T>>(number_of_cells);
        }
    }
    if (this->_fluid != nullptr
        && !this->_fluid->template has_state<atlas::fluid::FluidTemperatureState<T>>()) {
        this->_fluid->template emplace_state<atlas::fluid::FluidTemperatureState<T>>(this->_fluid->buffer_size());
    }
}

template <typename T>
void
BoltzmanMeasurer<T>::measure() {
    if (!this->make_probe()) {
        return;
    }
    const auto probe = this->_probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];
            Vector3<T> mean_velocity { T(0), T(0), T(0) };
            int count = 0;
            for (int k = begin; k < end; ++k) {
                const int particle_index = probe.indices_ptr[k];
                if (particle_index < 0 || particle_index >= probe.particle_count) continue;
                mean_velocity += probe.velocity_ptr[particle_index];
                ++count;
            }
            if (count <= 0) {
                probe.bulk_velocity_ptr[cell]     = Vector3<T> { T(0), T(0), T(0) };
                probe.thermal_energy_ptr[cell]    = T(0);
                probe.number_particle_ptr[cell]   = T(0);
                probe.field_temperature_ptr[cell] = T(0);
                return;
            }
            mean_velocity /= static_cast<T>(count);
            probe.bulk_velocity_ptr[cell] = mean_velocity;
            T thermal_energy_sum          = T(0);
            for (int k = begin; k < end; ++k) {
                const int particle_index = probe.indices_ptr[k];
                if (particle_index < 0 || particle_index >= probe.particle_count) continue;
                const Vector3<T> dv = probe.velocity_ptr[particle_index] - mean_velocity;
                thermal_energy_sum += dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
            }
            probe.thermal_energy_ptr[cell]    = thermal_energy_sum;
            probe.number_particle_ptr[cell]   = static_cast<T>(count);
            probe.field_temperature_ptr[cell] = thermal_energy_sum
                / (static_cast<T>(3)
                   * static_cast<T>(atlas::boltzmann_constant)
                   * static_cast<T>(count));
        });
    if (probe.particle_temperature_ptr != nullptr
        && (_measure_mode == MeasureModeType::Fluid || _measure_mode == MeasureModeType::All)) {
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            probe.num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                const int begin = probe.cell_start_ptr[cell];
                const int end   = probe.cell_end_ptr[cell];
                for (int k = begin; k < end; ++k) {
                    const int particle_index = probe.indices_ptr[k];
                    if (particle_index < 0 || particle_index >= probe.particle_count) continue;
                    probe.particle_temperature_ptr[particle_index] = probe.field_temperature_ptr[cell];
                }
            });
    }
}

template <typename T>
MeasureModeType
BoltzmanMeasurer<T>::measure_mode() const noexcept {
    return _measure_mode;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {
    _measure_mode = measure_mode;
    return *this;
}

template <typename T>
void
BoltzmanMeasurer<T>::Builder::validate() const {
    if (!_universe) {
        throw std::runtime_error("BoltzmanMeasurer::Builder: universe must not be null.");
    }
    if (!_fluid) {
        throw std::runtime_error("BoltzmanMeasurer::Builder: fluid must not be null.");
    }
    if (!_searcher) {
        throw std::runtime_error("BoltzmanMeasurer::Builder: searcher must not be null.");
    }
}

template <typename T>
BoltzmanMeasurer<T>
BoltzmanMeasurer<T>::Builder::build() const {
    validate();
    return BoltzmanMeasurer<T>(_universe, _fluid, _searcher, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<BoltzmanMeasurer<T>>
BoltzmanMeasurer<T>::Builder::make_host_shared() const {
    return atlas::make_host_shared<BoltzmanMeasurer<T>>(build());
}

}
