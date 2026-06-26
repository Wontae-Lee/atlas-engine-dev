#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas {

template <typename T>
typename BoltzmannMeasurer<T>::Builder
BoltzmannMeasurer<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
BoltzmannMeasurer<T>::BoltzmannMeasurer(UniverseHostPtr<T> universe,
                                      FluidHostPtr<T> fluid,
                                      SpatialHashingSearcherHostPtr<T> searcher,
                                      const MeasureModeType measure_mode) noexcept
    : Measurer<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _measure_mode(measure_mode) {
    if (this->_universe != nullptr) {

        const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

        if (!this->_universe->template has_state<atlas::UniverseTemperatureState<T>>()) {

            this->_universe->template emplace_state<atlas::UniverseTemperatureState<T>>(number_of_cells);
        }

        if (!this->_universe->template has_state<atlas::UniverseBulkVelocityState<T>>()) {

            this->_universe->template emplace_state<atlas::UniverseBulkVelocityState<T>>(number_of_cells);
        }

        if (!this->_universe->template has_state<atlas::UniverseThermalEnergyState<T>>()) {

            this->_universe->template emplace_state<atlas::UniverseThermalEnergyState<T>>(number_of_cells);
        }

        if (!this->_universe->template has_state<atlas::UniverseNumberParticleState<T>>()) {

            this->_universe->template emplace_state<atlas::UniverseNumberParticleState<T>>(number_of_cells);
        }
    }

    if (this->_fluid != nullptr
        && !this->_fluid->template has_state<atlas::FluidTemperatureState<T>>()) {

        this->_fluid->template emplace_state<atlas::FluidTemperatureState<T>>(this->_fluid->buffer_size());
    }
}

template <typename T>
void
BoltzmannMeasurer<T>::measure() {

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

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

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

            T thermal_energy_sum = T(0);

            for (int k = begin; k < end; ++k) {
                const int particle_index = probe.indices_ptr[k];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const Vector3<T> dv = probe.velocity_ptr[particle_index] - mean_velocity;

                thermal_energy_sum += dv.length_squared();
            }

            probe.thermal_energy_ptr[cell]  = thermal_energy_sum;
            probe.number_particle_ptr[cell] = static_cast<T>(count);

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

                    if (particle_index < 0 || particle_index >= probe.particle_count) {
                        continue;
                    }

                    probe.particle_temperature_ptr[particle_index] = probe.field_temperature_ptr[cell];
                }
            });
    }
}

template <typename T>
void
BoltzmannMeasurer<T>::measure(const T) {
    measure();
}

template <typename T>
MeasureModeType
BoltzmannMeasurer<T>::measure_mode() const noexcept {

    return _measure_mode;
}

template <typename T>
typename BoltzmannMeasurer<T>::Builder&
BoltzmannMeasurer<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {

    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename BoltzmannMeasurer<T>::Builder&
BoltzmannMeasurer<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename BoltzmannMeasurer<T>::Builder&
BoltzmannMeasurer<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename BoltzmannMeasurer<T>::Builder&
BoltzmannMeasurer<T>::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {

    _measure_mode = measure_mode;
    return *this;
}

template <typename T>
void
BoltzmannMeasurer<T>::Builder::validate() const {

    if (!_universe) {
        throw std::runtime_error("BoltzmannMeasurer::Builder: universe must not be null.");
    }

    if (!_fluid) {
        throw std::runtime_error("BoltzmannMeasurer::Builder: fluid must not be null.");
    }

    if (!_searcher) {
        throw std::runtime_error("BoltzmannMeasurer::Builder: searcher must not be null.");
    }
}

template <typename T>
BoltzmannMeasurer<T>
BoltzmannMeasurer<T>::Builder::build() const {

    validate();

    return BoltzmannMeasurer<T>(_universe, _fluid, _searcher, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<BoltzmannMeasurer<T>>
BoltzmannMeasurer<T>::Builder::make_host_shared() const {

    return atlas::make_host_shared<BoltzmannMeasurer<T>>(build());
}

}