#pragma once

#include <atlas/core/constants.h>
#include <atlas/logging/logging.h>
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
    : Measure<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _measure_mode(measure_mode) {
}

template <typename T>
void
BoltzmanMeasurer<T>::measure() {

    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return;
    }

    auto* universe_temperature
        = this->_universe->template state<atlas::universe::UniverseTemperatureState<T>>();
    auto* universe_bulk_velocity
        = this->_universe->template state<atlas::universe::UniverseBulkVelocityState<T>>();
    auto* universe_momentum_weight
        = this->_universe->template state<atlas::universe::UniverseMomentumWeightState<T>>();
    auto* universe_thermal_energy
        = this->_universe->template state<atlas::universe::UniverseThermalEnergyState<T>>();

    auto* fluid_velocity = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* fluid_temperature = this->_fluid->template state<atlas::fluid::FluidTemperatureState<T>>();

    if (universe_temperature == nullptr || universe_bulk_velocity == nullptr
        || universe_momentum_weight == nullptr || universe_thermal_energy == nullptr
        || fluid_velocity == nullptr) {
        return;
    }

    auto& field_temperature = universe_temperature->temperature;
    auto& bulk_velocity = universe_bulk_velocity->bulk_velocity;
    auto& momentum_weight = universe_momentum_weight->momentum_weight;
    auto& thermal_energy = universe_thermal_energy->thermal_energy;
    auto& particle_velocity = fluid_velocity->data();

    auto* field_temperature_ptr = atlas::raw_pointer_cast(field_temperature.data());
    auto* bulk_velocity_ptr = atlas::raw_pointer_cast(bulk_velocity.data());
    auto* momentum_weight_ptr = atlas::raw_pointer_cast(momentum_weight.data());
    auto* thermal_energy_ptr = atlas::raw_pointer_cast(thermal_energy.data());
    auto* particle_temperature_ptr = fluid_temperature != nullptr
        ? atlas::raw_pointer_cast(fluid_temperature->data().data())
        : nullptr;
    const auto* velocity_ptr = atlas::raw_pointer_cast(particle_velocity.data());
    const auto* indices_ptr = this->_searcher->indices();
    const auto* cell_start_ptr = this->_searcher->cell_start();
    const auto* cell_end_ptr = this->_searcher->cell_end();
    const int particle_count = static_cast<int>(this->_fluid->particle_count());
    const auto num_of_cells = this->_universe->number_of_cells();

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = cell_start_ptr[cell];

            const int end = cell_end_ptr[cell];

            Vector3<T> mean_velocity { T(0), T(0), T(0) };

            T momentum_weight_sum = T(0);

            int count = 0;

            for (int k = begin; k < end; ++k) {

                const int particle_index = indices_ptr[k];

                if (particle_index < 0 || particle_index >= particle_count) continue;

                mean_velocity += velocity_ptr[particle_index];
                momentum_weight_sum += T(1);

                ++count;
            }

            if (count <= 0 || !(momentum_weight_sum > T(0))) {
                bulk_velocity_ptr[cell] = Vector3<T> { T(0), T(0), T(0) };
                momentum_weight_ptr[cell] = T(0);
                thermal_energy_ptr[cell] = T(0);
                field_temperature_ptr[cell] = T(0);
                return;
            }

            mean_velocity /= momentum_weight_sum;
            bulk_velocity_ptr[cell] = mean_velocity;
            momentum_weight_ptr[cell] = momentum_weight_sum;

            T thermal_energy_sum = T(0);

            for (int k = begin; k < end; ++k) {

                const int particle_index = indices_ptr[k];

                if (particle_index < 0 || particle_index >= particle_count) continue;

                const Vector3<T> dv = velocity_ptr[particle_index] - mean_velocity;

                thermal_energy_sum += dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
            }

            thermal_energy_ptr[cell] = thermal_energy_sum;
            field_temperature_ptr[cell] = thermal_energy_sum
                / (static_cast<T>(3)
                   * static_cast<T>(atlas::boltzmann_constant)
                   * static_cast<T>(count));
        });

    if (particle_temperature_ptr != nullptr
        && (_measure_mode == MeasureModeType::Fluid || _measure_mode == MeasureModeType::All)) {

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                const int begin = cell_start_ptr[cell];

                const int end = cell_end_ptr[cell];

                for (int k = begin; k < end; ++k) {

                    const int particle_index = indices_ptr[k];
                    if (particle_index < 0 || particle_index >= particle_count) continue;

                    particle_temperature_ptr[particle_index] = field_temperature_ptr[cell];
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
