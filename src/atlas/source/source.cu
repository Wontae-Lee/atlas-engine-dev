#include <atlas/fluid/fluid_state.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/source/source.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace atlas {

Source::Source(UniverseHostPtr universe,
               DeviceBuffer<SpawnType> spawn_types,
               DeviceBuffer<Spawn> spawn_operators,
               FluidHostPtr fluid,
               const bool flip,
               const float spacing,
               const float tolerance,
               const float temperature,
               ObserverHostPtr observer) noexcept
    : _universe(std::move(universe))
    , _spawn_types(std::move(spawn_types))
    , _spawn_operators(std::move(spawn_operators))
    , _fluid(std::move(fluid))
    , _observer(std::move(observer))
    , _flip(flip)
    , _spacing(spacing)
    , _tolerance(tolerance)
    , _temperature(temperature)
    , _is_invalidated_cache(true) {
}

Source::Builder
Source::builder() noexcept {

    return Builder {};
}

void
Source::update(const float dt) {

    if (!_universe || _universe->source_units().empty() || !(dt > 0.0f)) {
        return;
    }

    _universe->source_units().advance(dt);

    emit();
}

void
Source::rebuild_cache() noexcept {

    if (!_is_invalidated_cache) {
        return;
    }

    _cache_builder.rebuild(_universe->source_units().units(),
                           _spawn_types,
                           _spawn_operators,
                           _fluid,
                           _flip,
                           _spacing,
                           _tolerance,
                           _local_unit_counts,
                           _flat_local_positions,
                           _flat_unit_indices,
                           _local_particle_count,
                           _species_cache,
                           _shuffled_species,
                           _shuffle_keys,
                           _shuffle_seed);

    _is_invalidated_cache = false;
}

void
Source::shuffle_species(const std::size_t count) {
    _species_shuffler.shuffle(_species_cache, _shuffled_species, _shuffle_keys, _shuffle_seed, count);
}

void
Source::emit() {

    const std::size_t step_index = _step_index++;

    auto* source_sensor_metrics = _observer ? _observer->sensor_metrics<atlas::SourceSensorMetrics>() : nullptr;

    HostBuffer<std::size_t> emitted_per_unit;

    if (source_sensor_metrics != nullptr) {

        emitted_per_unit = HostBuffer<std::size_t>(_universe->source_units().size(), std::size_t { 0 });
    }

    const auto record_source_metrics = [&] {
        if (source_sensor_metrics == nullptr) {
            return;
        }

        for (std::size_t unit_index = 0; unit_index < emitted_per_unit.size(); ++unit_index) {
            source_sensor_metrics->record(step_index, unit_index, emitted_per_unit[unit_index]);
        }
    };

    if (!_fluid) {

        record_source_metrics();
        return;
    }

    rebuild_cache();

    if (_local_particle_count == 0) {

        record_source_metrics();
        return;
    }

    const std::size_t current_particle_count = _fluid->particle_count();

    // Fluid has a fixed capacity (buffer_size()); emission can never exceed
    // the remaining slots regardless of how many candidate positions the
    // cache offers this step.
    const std::size_t available_slots = current_particle_count < _fluid->buffer_size()
        ? (_fluid->buffer_size() - current_particle_count)
        : std::size_t { 0 };

    const std::size_t emit_count = _local_particle_count < available_slots
        ? _local_particle_count
        : available_slots;

    if (emit_count == 0) {

        atlas::warn() << "\n"
                      << "Source emission skipped: no available slots for "
                      << _local_particle_count
                      << " particles\n";

        record_source_metrics();
        return;
    }

    shuffle_species(emit_count);

    if (!make_probe()) {
        record_source_metrics();
        return;
    }

    const auto probe = _probe;

    if (source_sensor_metrics != nullptr) {
        int offset = 0;
        for (std::size_t u = 0; u < _local_unit_counts.size(); ++u) {
            const int unit_size     = _local_unit_counts[u];
            const std::size_t start = static_cast<std::size_t>(offset);
            const std::size_t end   = start + static_cast<std::size_t>(unit_size);
            if (emit_count > start) {
                emitted_per_unit[u] = std::min(emit_count, end) - start;
            }
            offset += unit_size;
        }
    }

    _emitter.emit(probe, current_particle_count, emit_count);

    _fluid->set_particle_count(current_particle_count + emit_count);

    record_source_metrics();
}

bool
Source::make_probe() noexcept {
    _probe = {};

    const auto& units = _universe->source_units().units();

    if (!_fluid || units.empty() || _shuffled_species.empty()) {
        return false;
    }

    auto* position_state = _fluid->state<FluidPositionState>();
    auto* velocity_state = _fluid->state<FluidVelocityState>();
    auto* species_state  = _fluid->state<FluidSpeciesState>();
    auto* active_state   = _fluid->state<FluidActiveState>();

    if (position_state == nullptr || velocity_state == nullptr || species_state == nullptr || active_state == nullptr) {
        return false;
    }

    auto& positions_buf        = position_state->data();
    auto& velocities_buf       = velocity_state->data();
    auto& species_buf          = species_state->data();
    auto& active_buf           = active_state->data();
    const auto& generators_buf = _fluid->generators();
    const auto& properties_buf = _fluid->particle_properties();

    if (positions_buf.empty() || velocities_buf.empty() || species_buf.empty() || active_buf.empty()
        || generators_buf.empty() || properties_buf.empty()
        || _flat_local_positions.empty() || _flat_unit_indices.empty()) {
        return false;
    }

    _probe.units                = atlas::raw_pointer_cast(units.data());
    _probe.generators           = atlas::raw_pointer_cast(generators_buf.data());
    _probe.properties           = atlas::raw_pointer_cast(properties_buf.data());
    _probe.shuffled_species     = atlas::raw_pointer_cast(_shuffled_species.data());
    _probe.positions            = atlas::raw_pointer_cast(positions_buf.data());
    _probe.velocities           = atlas::raw_pointer_cast(velocities_buf.data());
    _probe.species              = atlas::raw_pointer_cast(species_buf.data());
    _probe.active               = atlas::raw_pointer_cast(active_buf.data());
    _probe.flat_local_positions = atlas::raw_pointer_cast(_flat_local_positions.data());
    _probe.flat_unit_indices    = atlas::raw_pointer_cast(_flat_unit_indices.data());
    _probe.temperature          = _temperature;
    _probe.property_count       = static_cast<int>(properties_buf.size());
    _probe.emission_seed        = _shuffle_seed;

    return true;
}

Source
Source::Builder::build() {

    validate();

    return Source(
        _universe,
        DeviceBuffer<SpawnType>(_spawn_types.begin(), _spawn_types.end()),
        DeviceBuffer<Spawn>(_spawn_operators.begin(), _spawn_operators.end()),
        _fluid,
        _flip,
        _spacing,
        _tolerance,
        _temperature,
        _observer);
}

atlas::host_shared_ptr<Source>
Source::Builder::make_host_shared() {

    return atlas::make_host_shared<Source>(build());
}

Source::Builder&
Source::Builder::with_universe(UniverseHostPtr universe) noexcept {

    _universe = std::move(universe);

    return *this;
}

Source::Builder&
Source::Builder::with_fluid(FluidHostPtr fluid) noexcept {

    _fluid = std::move(fluid);
    return *this;
}

Source::Builder&
Source::Builder::with_observer(ObserverHostPtr observer) noexcept {

    _observer = std::move(observer);
    return *this;
}

Source::Builder&
Source::Builder::with_spawn_types(const HostBuffer<SpawnType>& spawn_types) {

    if (spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    _spawn_types.insert(_spawn_types.end(), spawn_types.begin(), spawn_types.end());

    return *this;
}

Source::Builder&
Source::Builder::with_spawn_operator(const Spawn& spawn_operator) noexcept {

    _spawn_operators.push_back(spawn_operator);
    return *this;
}

Source::Builder&
Source::Builder::with_spawn_operators(const HostBuffer<Spawn>& spawn_operators) {

    if (spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    _spawn_operators.insert(_spawn_operators.end(), spawn_operators.begin(), spawn_operators.end());

    return *this;
}

Source::Builder&
Source::Builder::with_tolerance(const float tolerance) noexcept {

    _tolerance = tolerance;
    return *this;
}

Source::Builder&
Source::Builder::with_flip(const bool flip) noexcept {

    _flip = flip;
    return *this;
}

Source::Builder&
Source::Builder::with_spacing(const float spacing) noexcept {

    _spacing = spacing;
    return *this;
}

Source::Builder&
Source::Builder::with_temperature(const float temperature) noexcept {

    _temperature = temperature;
    return *this;
}

void
Source::Builder::validate() const {

    if (!_universe) {
        throw std::runtime_error("Source::Builder: universe must not be null.");
    }

    const std::size_t unit_count = _universe->source_units().size();

    if (unit_count == 0) {
        throw std::runtime_error("Source::Builder: universe must hold at least one source unit.");
    }

    if (!_fluid) {
        throw std::runtime_error("Source::Builder: fluid must be provided.");
    }

    if (_spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    if (_spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    if (_spawn_types.size() != 1 && _spawn_types.size() != unit_count) {
        throw std::runtime_error(
            "Source::Builder: spawn types must have size 1 or match unit count.");
    }

    if (_spawn_operators.size() != 1 && _spawn_operators.size() != unit_count) {
        throw std::runtime_error(
            "Source::Builder: spawn operators must have size 1 or match unit count.");
    }

    if (!atlas::isfinite(_spacing) || _spacing <= 0.0f) {
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    if (!atlas::isfinite(_tolerance)) {
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    if (!atlas::isfinite(_temperature)) {
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

}
