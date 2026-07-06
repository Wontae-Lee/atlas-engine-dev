#include <atlas/measure/boltzmann_measurer.h>

#include <atlas/math/math.h>
#include <atlas/parallel/parallel_for.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

BoltzmannMeasurer::Builder
BoltzmannMeasurer::builder() noexcept {

    return Builder {};
}

BoltzmannMeasurer::BoltzmannMeasurer(UniverseHostPtr universe,
                                     FluidHostPtr fluid,
                                     SearcherHostPtr searcher,
                                     const MeasureModeType measure_mode) noexcept
    : Measurer(std::move(universe), std::move(fluid), std::move(searcher))
    , _measure_mode(measure_mode) {
    if (_universe != nullptr) {

        const auto cell_count = static_cast<std::size_t>(_universe->cell_count());

        if (!_universe->has_state<atlas::UniverseTemperatureState>()) {

            _universe->emplace_state<atlas::UniverseTemperatureState>(cell_count);
        }

        if (!_universe->has_state<atlas::UniverseBulkVelocityState>()) {

            _universe->emplace_state<atlas::UniverseBulkVelocityState>(cell_count);
        }

        if (!_universe->has_state<atlas::UniverseThermalEnergyState>()) {

            _universe->emplace_state<atlas::UniverseThermalEnergyState>(cell_count);
        }

        if (!_universe->has_state<atlas::UniverseNumberParticleState>()) {

            _universe->emplace_state<atlas::UniverseNumberParticleState>(cell_count);
        }
    }

    if (_fluid != nullptr
        && !_fluid->has_state<atlas::FluidTemperatureState>()) {

        _fluid->emplace_state<atlas::FluidTemperatureState>(_fluid->buffer_size());
    }
}

void
BoltzmannMeasurer::measure() {

    if (!make_probe()) {
        return;
    }

    measure_field();

    if (_probe.particle_temperature_ptr != nullptr
        && (_measure_mode == MeasureModeType::fluid || _measure_mode == MeasureModeType::all)) {
        assign_particle_temperature();
    }
}

void
BoltzmannMeasurer::measure(const float) {
    measure();
}

MeasureModeType
BoltzmannMeasurer::measure_mode() const noexcept {

    return _measure_mode;
}

void
BoltzmannMeasurer::measure_field() {

    const auto probe = _probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
            const int begin = probe.cell_start_ptr[cell];
            const int end   = probe.cell_end_ptr[cell];

            // Two passes are required: the bulk (mean) velocity must be
            // fully known before the second pass can measure each
            // particle's *fluctuation* about it — computing them in one
            // pass would conflate bulk flow with thermal motion.
            Float3 mean_velocity(0.0f, 0.0f, 0.0f);
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

                probe.bulk_velocity_ptr[cell]     = Float3(0.0f, 0.0f, 0.0f);
                probe.thermal_energy_ptr[cell]    = 0.0f;
                probe.number_particle_ptr[cell]   = 0.0f;
                probe.field_temperature_ptr[cell] = 0.0f;
                return;
            }

            mean_velocity /= static_cast<float>(count);

            probe.bulk_velocity_ptr[cell] = mean_velocity;

            float thermal_energy_sum = 0.0f;

            for (int k = begin; k < end; ++k) {
                const int particle_index = probe.indices_ptr[k];

                if (particle_index < 0 || particle_index >= probe.particle_count) {
                    continue;
                }

                const Float3 dv = probe.velocity_ptr[particle_index] - mean_velocity;

                thermal_energy_sum += dv.length_squared();
            }

            probe.thermal_energy_ptr[cell]  = thermal_energy_sum;
            probe.number_particle_ptr[cell] = static_cast<float>(count);

            // Equipartition theorem: (3/2) k_B T = (1/2) <|v-u|^2>, so
            // T = <|v-u|^2> / (3 k_B). No molecular-mass factor appears —
            // this implicitly assumes unit mass (see boltzmann_measurer.h's
            // top-of-file documentation for why).
            probe.field_temperature_ptr[cell] = thermal_energy_sum
                / (3.0f
                   * atlas::boltzmann_constant
                   * static_cast<float>(count));
        });
}

void
BoltzmannMeasurer::assign_particle_temperature() {

    const auto probe = _probe;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        probe.cell_count,
        [=] ATLAS_ALL_DEVICE(const int cell) {
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

BoltzmannMeasurer::Builder&
BoltzmannMeasurer::Builder::with_universe(UniverseHostPtr universe) noexcept {

    _universe = std::move(universe);
    return *this;
}

BoltzmannMeasurer::Builder&
BoltzmannMeasurer::Builder::with_fluid(FluidHostPtr fluid) noexcept {

    _fluid = std::move(fluid);
    return *this;
}

BoltzmannMeasurer::Builder&
BoltzmannMeasurer::Builder::with_searcher(SearcherHostPtr searcher) noexcept {

    _searcher = std::move(searcher);
    return *this;
}

BoltzmannMeasurer::Builder&
BoltzmannMeasurer::Builder::with_measure_mode(const MeasureModeType measure_mode) noexcept {

    _measure_mode = measure_mode;
    return *this;
}

void
BoltzmannMeasurer::Builder::validate() const {

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

BoltzmannMeasurer
BoltzmannMeasurer::Builder::build() const {

    validate();

    return BoltzmannMeasurer(_universe, _fluid, _searcher, _measure_mode);
}

atlas::host_shared_ptr<BoltzmannMeasurer>
BoltzmannMeasurer::Builder::make_host_shared() const {

    return atlas::make_host_shared<BoltzmannMeasurer>(build());
}

}
