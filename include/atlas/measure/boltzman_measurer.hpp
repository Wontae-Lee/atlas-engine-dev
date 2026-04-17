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

    // Return a default-initialized builder for fluent measurer construction.
    return Builder {};
}

template <typename T>
BoltzmanMeasurer<T>::BoltzmanMeasurer(UniverseHostPtr<T> universe,
                                      FluidHostPtr<T> fluid,
                                      SpatialHashingSearcherHostPtr<T> searcher,
                                      const MeasureModeType measure_mode) noexcept
    : Measure<T>(std::move(universe), std::move(fluid), std::move(searcher))
    , _measure_mode(measure_mode) {
    // Store the measurement mode together with the inherited universe/fluid/searcher dependencies.

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

    // A valid measurement pass requires all three core dependencies:
    // - universe : receives cell-based measured fields
    // - fluid    : provides particle-based input states
    // - searcher : provides the cell-to-particle mapping
    //
    // If any dependency is missing, there is no consistent way to perform
    // the measurement, so return without modifying anything.
    if (!this->_universe || !this->_fluid || !this->_searcher) {
        return;
    }

    // Retrieve the universe-side output states that will store the measured
    // macroscopic quantities for each cell.
    //
    // These states represent:
    // - temperature      : cell temperature
    // - bulk velocity    : average particle velocity in the cell
    // - number particle  : number of particles assigned to the cell
    // - thermal energy   : sum of squared thermal velocity fluctuations
    auto* universe_temperature
        = this->_universe->template state<atlas::universe::UniverseTemperatureState<T>>();
    auto* universe_bulk_velocity
        = this->_universe->template state<atlas::universe::UniverseBulkVelocityState<T>>();
    auto* universe_thermal_energy
        = this->_universe->template state<atlas::universe::UniverseThermalEnergyState<T>>();
    auto* universe_number_particle
        = this->_universe->template state<atlas::universe::UniverseNumberParticleState<T>>();

    // Retrieve the fluid-side states used during measurement.
    //
    // Velocity is the required particle input.
    // Temperature is optional and is only written when:
    // - the state exists, and
    // - the measure mode requests fluid-side output.
    auto* fluid_velocity    = this->_fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* fluid_temperature = this->_fluid->template state<atlas::fluid::FluidTemperatureState<T>>();

    // The measurement requires all universe-side output states and the
    // fluid velocity input state.
    //
    // If any mandatory state is missing, stop early rather than partially
    // updating only some outputs.
    if (universe_temperature == nullptr || universe_bulk_velocity == nullptr
        || universe_thermal_energy == nullptr || universe_number_particle == nullptr
        || fluid_velocity == nullptr) {
        return;
    }

    // Bind references to the underlying field buffers for readability.
    auto& field_temperature = universe_temperature->data();
    auto& bulk_velocity     = universe_bulk_velocity->data();
    auto& thermal_energy    = universe_thermal_energy->data();
    auto& number_particle   = universe_number_particle->data();
    auto& particle_velocity = fluid_velocity->data();

    // Convert all container-backed storage into raw pointers so the device
    // kernels can access them directly.
    //
    // Searcher-provided arrays define the particle index interval for each cell:
    //   [cell_start[cell], cell_end[cell])
    // over the flattened particle-index array `indices_ptr`.
    auto* field_temperature_ptr    = atlas::raw_pointer_cast(field_temperature.data());
    auto* bulk_velocity_ptr        = atlas::raw_pointer_cast(bulk_velocity.data());
    auto* thermal_energy_ptr       = atlas::raw_pointer_cast(thermal_energy.data());
    auto* number_particle_ptr      = atlas::raw_pointer_cast(number_particle.data());
    auto* particle_temperature_ptr = fluid_temperature != nullptr
        ? atlas::raw_pointer_cast(fluid_temperature->data().data())
        : nullptr;
    const auto* velocity_ptr       = atlas::raw_pointer_cast(particle_velocity.data());
    const auto* indices_ptr        = this->_searcher->indices();
    const auto* cell_start_ptr     = this->_searcher->cell_start();
    const auto* cell_end_ptr       = this->_searcher->cell_end();
    const int particle_count       = static_cast<int>(this->_fluid->particle_count());
    const auto num_of_cells        = this->_universe->number_of_cells();
    // First pass:
    // Compute cell-wise macroscopic quantities from the particle velocities.
    //
    // For each cell:
    // 1. accumulate particle velocities
    // 2. compute mean velocity = bulk velocity
    // 3. accumulate squared deviations from the mean
    // 4. convert that fluctuation measure into thermal energy and temperature
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Get the half-open interval of particle references belonging to
            // this cell inside the flattened particle-index list.
            const int begin = cell_start_ptr[cell];
            const int end   = cell_end_ptr[cell];

            // This variable initially accumulates the sum of particle velocities.
            // After normalization, it becomes the cell bulk velocity.
            Vector3<T> mean_velocity { T(0), T(0), T(0) };

            // Count the number of valid particles contributing to this cell.
            int count = 0;

            // First accumulation loop:
            // Sum particle velocities to obtain the mean/bulk velocity later.
            for (int k = begin; k < end; ++k) {

                const int particle_index = indices_ptr[k];

                // Skip invalid particle references defensively.
                if (particle_index < 0 || particle_index >= particle_count) continue;

                mean_velocity += velocity_ptr[particle_index];
                ++count;
            }

            // If the cell contains no valid particles, explicitly reset all
            // output quantities to zero so stale values do not remain.
            if (count <= 0) {
                bulk_velocity_ptr[cell]     = Vector3<T> { T(0), T(0), T(0) };
                thermal_energy_ptr[cell]    = T(0);
                number_particle_ptr[cell]   = T(0);
                field_temperature_ptr[cell] = T(0);
                return;
            }

            // Normalize the accumulated velocity sum to obtain the cell bulk velocity.
            //
            // Physically, this is the average particle velocity in the cell and
            // represents the macroscopic drift motion.
            mean_velocity /= static_cast<T>(count);
            bulk_velocity_ptr[cell]   = mean_velocity;

            T thermal_energy_sum = T(0);

            // Second accumulation loop:
            // Measure thermal motion relative to the bulk velocity.
            //
            // For each particle, subtract the mean velocity to isolate the
            // fluctuating part of motion:
            //   dv = v_i - mean_velocity
            //
            // The sum of squared magnitudes of these fluctuations is used here
            // as the cell thermal-energy-like quantity.
            for (int k = begin; k < end; ++k) {

                const int particle_index = indices_ptr[k];

                // Again skip invalid particle references defensively.
                if (particle_index < 0 || particle_index >= particle_count) continue;

                const Vector3<T> dv = velocity_ptr[particle_index] - mean_velocity;

                thermal_energy_sum += dv.x * dv.x + dv.y * dv.y + dv.z * dv.z;
            }

            // Store the accumulated fluctuation energy proxy for the cell.
            thermal_energy_ptr[cell] = thermal_energy_sum;

            // Store the number of valid particles assigned to this cell.
            number_particle_ptr[cell] = static_cast<T>(count);

            // Convert the fluctuation sum into temperature using a Boltzmann-style relation.
            //
            // The formula used here is:
            //   T_cell = thermal_energy_sum / (3 * k_B * count)
            //
            // Interpretation:
            // - 3      : three translational degrees of freedom
            // - k_B    : Boltzmann constant
            // - count  : average per particle
            //
            // This implementation effectively assumes normalized particle mass
            // and a velocity-based thermal-energy formulation consistent with
            // the project's unit system.
            field_temperature_ptr[cell] = thermal_energy_sum
                / (static_cast<T>(3)
                   * static_cast<T>(atlas::boltzmann_constant)
                   * static_cast<T>(count));
        });

    // Optional second pass:
    // Write the measured cell temperature back to particle temperature state.
    //
    // This happens only when:
    // - the fluid temperature state exists, and
    // - the measure mode requests fluid-side output
    //
    // Each particle inherits the temperature of the cell it belongs to.
    if (particle_temperature_ptr != nullptr
        && (_measure_mode == MeasureModeType::Fluid || _measure_mode == MeasureModeType::All)) {

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                const int begin = cell_start_ptr[cell];
                const int end   = cell_end_ptr[cell];

                // Assign the already-computed cell temperature to every valid
                // particle listed in this cell.
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

    // Return the configured measure mode.
    return _measure_mode;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_universe(UniverseHostPtr<T> universe) noexcept {

    // Store the target universe for later construction.
    _universe = std::move(universe);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {

    // Store the target fluid for later construction.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_searcher(SpatialHashingSearcherHostPtr<T> searcher) noexcept {

    // Store the searcher providing cell-particle mapping.
    _searcher = std::move(searcher);
    return *this;
}

template <typename T>
typename BoltzmanMeasurer<T>::Builder&
BoltzmanMeasurer<T>::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {

    // Store the output application mode.
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

    // Validate builder dependencies before constructing the measurer.
    validate();

    return BoltzmanMeasurer<T>(_universe, _fluid, _searcher, _measure_mode);
}

template <typename T>
atlas::host_shared_ptr<BoltzmanMeasurer<T>>
BoltzmanMeasurer<T>::Builder::make_host_shared() const {

    // Construct a shared host-side measurer instance.
    return atlas::make_host_shared<BoltzmanMeasurer<T>>(build());
}

} // namespace atlas::system
