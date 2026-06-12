#pragma once

#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <stdexcept>

namespace atlas {

template <typename T>
typename BoltzmanMeasurer<T>::Builder
BoltzmanMeasurer<T>::builder() noexcept {
    // Return a fresh builder for fluent BoltzmanMeasurer construction.
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
        // Match universe field-state sizes to the current number of spatial cells.
        const auto number_of_cells = static_cast<std::size_t>(this->_universe->number_of_cells());

        if (!this->_universe->template has_state<atlas::UniverseTemperatureState<T>>()) {
            // Store per-cell temperature measurements in the universe state container.
            this->_universe->template emplace_state<atlas::UniverseTemperatureState<T>>(number_of_cells);
        }

        if (!this->_universe->template has_state<atlas::UniverseBulkVelocityState<T>>()) {
            // Store per-cell mean particle velocity.
            this->_universe->template emplace_state<atlas::UniverseBulkVelocityState<T>>(number_of_cells);
        }

        if (!this->_universe->template has_state<atlas::UniverseThermalEnergyState<T>>()) {
            // Store per-cell accumulated thermal velocity fluctuation energy.
            this->_universe->template emplace_state<atlas::UniverseThermalEnergyState<T>>(number_of_cells);
        }

        if (!this->_universe->template has_state<atlas::UniverseNumberParticleState<T>>()) {
            // Store the number of particles assigned to each spatial cell.
            this->_universe->template emplace_state<atlas::UniverseNumberParticleState<T>>(number_of_cells);
        }
    }

    if (this->_fluid != nullptr
        && !this->_fluid->template has_state<atlas::FluidTemperatureState<T>>()) {
        // Store per-particle temperature values when fluid-side temperature data is absent.
        this->_fluid->template emplace_state<atlas::FluidTemperatureState<T>>(this->_fluid->buffer_size());
    }
}

template <typename T>
void
BoltzmanMeasurer<T>::measure() {
    // Abort measurement if the required probe data cannot be assembled.
    if (!this->make_probe()) {
        return;
    }

    // Copy the probe descriptor so it can be captured by the device lambda.
    const auto probe = this->_probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            // Accumulate particle velocity to compute the cell bulk velocity.
            Vector3<T> mean_velocity { T(0), T(0), T(0) };
            int count = 0;

            for (int k = begin; k < end; ++k) {
                const int particle_index = probe.indices_ptr[k];

                // Skip invalid particle indices defensively.
                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                mean_velocity += probe.velocity_ptr[particle_index];
                ++count;
            }

            if (count <= 0) {
                // Empty cells receive zero-valued measured fields.
                probe.bulk_velocity_ptr[cell]     = Vector3<T> { T(0), T(0), T(0) };
                probe.thermal_energy_ptr[cell]    = T(0);
                probe.number_particle_ptr[cell]   = T(0);
                probe.field_temperature_ptr[cell] = T(0);
                return;
            }

            // Convert velocity sum into arithmetic mean velocity.
            mean_velocity /= static_cast<T>(count);

            // Store the cell bulk velocity.
            probe.bulk_velocity_ptr[cell] = mean_velocity;

            T thermal_energy_sum = T(0);

            for (int k = begin; k < end; ++k) {
                const int particle_index = probe.indices_ptr[k];

                // Skip invalid particle indices defensively.
                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                // Thermal fluctuation velocity is measured relative to the cell mean velocity.
                const Vector3<T> dv = probe.velocity_ptr[particle_index] - mean_velocity;

                // Accumulate squared fluctuation speed.
                thermal_energy_sum += dv.length_squared();
            }

            // Store raw thermal energy proxy and particle count for this cell.
            probe.thermal_energy_ptr[cell]  = thermal_energy_sum;
            probe.number_particle_ptr[cell] = static_cast<T>(count);

            // Convert mean fluctuation energy into a temperature-like field using 3 k_B N.
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

                    // Skip invalid particle indices defensively.
                    if (particle_index < 0 || particle_index >= probe.particle_count) {
                        continue;
                    }

                    // Assign the measured cell temperature back to each particle in that cell.
                    probe.particle_temperature_ptr[particle_index] = probe.field_temperature_ptr[cell];
                }
            });
    }
}

template <typename T>
void
BoltzmanMeasurer<T>::measure(const T) {
    measure();
}

template <typename T>
MeasureModeType
BoltzmanMeasurer<T>::measure_mode() const noexcept {
    // Return the configured target mode for temperature measurement output.
    return _measure_mode;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {
    // Store the universe object used for cell-level measurement states.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Store the fluid object used for particle-level velocity and temperature data.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {
    // Store the spatial searcher used to map particles into cells.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {
    // Store whether measured temperature is written to fluid, universe, or both.
    _measure_mode = measure_mode;
    return *this;
}

template <typename T>
void
BoltzmanMeasurer<T>::Builder::validate() const {
    // A universe is required because cell-level states are measured and updated.
    if (!_universe) {
        throw std::runtime_error("BoltzmanMeasurer::Builder: universe must not be null.");
    }

    // A fluid is required because particle velocities are the source measurement data.
    if (!_fluid) {
        throw std::runtime_error("BoltzmanMeasurer::Builder: fluid must not be null.");
    }

    // A searcher is required to provide particle-to-cell membership information.
    if (!_searcher) {
        throw std::runtime_error("BoltzmanMeasurer::Builder: searcher must not be null.");
    }
}

template <typename T>
BoltzmanMeasurer<T>
BoltzmanMeasurer<T>::Builder::build() const {
    // Validate required dependencies before constructing the measurer.
    validate();

    return BoltzmanMeasurer<T>(_universe, _fluid, _searcher, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<BoltzmanMeasurer<T>>
BoltzmanMeasurer<T>::Builder::make_host_shared() const {
    // Build a validated measurer and store it in host-managed shared ownership.
    return atlas::make_host_shared<BoltzmanMeasurer<T>>(build());
}

} // namespace atlas
