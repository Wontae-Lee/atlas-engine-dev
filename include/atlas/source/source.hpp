#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/sampling/sampling.h>
#include <atlas/shuffle/shuffle_operator.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::fluid {

template <typename T>
Source<T>::Source(DeviceBuffer<Unit<T>> units,
                  DeviceBuffer<SpawnType> spawn_types,
                  DeviceBuffer<SpawnOperator<T>> spawn_operators,
                  atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
                  const bool flip,
                  const T spacing,
                  const T tolerance,
                  const T temperature,
                  ObserverHostPtr observer) noexcept
    : _units(std::move(units))
    , _spawn_types(std::move(spawn_types))
    , _spawn_operators(std::move(spawn_operators))
    , _fluid(std::move(fluid))
    , _observer(std::move(observer))
    , _flip(flip)
    , _spacing(spacing)
    , _tolerance(tolerance)
    , _temperature(temperature)
    , _is_invalidated_cache(true) {
    // Store all source configuration and mark the cache as invalid.
    //
    // The actual spawn-position cache is built lazily on demand in emit()
    // through rebuild_cache().
}

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {

    // Return a default-initialized builder for fluent Source construction.
    return Builder {};
}

template <typename T>
void
Source<T>::update(const T dt) {

    // No update is needed when there are no units or the time step is invalid.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());

    // Advance every source unit independently on the device.
    //
    // A unit may internally update its transform, activation schedule,
    // animation state, or other time-dependent emission properties.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });

    emit();
}

template <typename T>
void
Source<T>::rebuild_cache() noexcept {

    // Reuse the cached local emission layout until the source configuration
    // changes and explicitly invalidates it.
    if (!_is_invalidated_cache) {
        return;
    }

    // If the source is not in a usable state, clear every cache-dependent buffer
    // and leave the cache in a valid empty state.
    if (_units.empty() || _spawn_types.empty() || _spawn_operators.empty() || !_fluid || _fluid->generators().empty()) {
        _local_positions.clear();
        _local_particle_count = 0;
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    const auto& generators = _fluid->generators();

    // Copy units and spawn operators to host memory because cache generation
    // below is performed with host-side iteration over bounds and candidate samples.
    const HostBuffer<Unit<T>> units(_units.begin(), _units.end());
    const HostBuffer<SpawnOperator<T>> spawn_operators(
        _spawn_operators.begin(),
        _spawn_operators.end());

    _local_positions.clear();
    _local_positions.resize(units.size());

    std::size_t total_count                = 0;
    const std::size_t spawn_operator_count = spawn_operators.size();

    // Build a local candidate-position list for each source unit.
    //
    // The positions are stored in the unit's local space and transformed into
    // world space later during emit().
    for (std::size_t i = 0; i < units.size(); ++i) {
        const auto& geometry = units[i].geometry_operator();

        const auto bounds = geometry.bound();

        const auto& lower = bounds.lower_corner;
        const auto& upper = bounds.upper_corner;

        // Support either:
        // - one shared spawn operator for all units, or
        // - one spawn operator per unit
        const auto& spawn_op = spawn_operators[(spawn_operator_count == 1) ? 0 : i];

        // Determine how many regularly spaced samples fit along each axis of the
        // unit's bounding box.
        const int nx = atlas::sampling::sample_axis_count(lower.x, upper.x, _spacing);
        const int ny = atlas::sampling::sample_axis_count(lower.y, upper.y, _spacing);
        const int nz = atlas::sampling::sample_axis_count(lower.z, upper.z, _spacing);

        HostBuffer<Vector3<T>> positions;

        if (nx > 0 && ny > 0 && nz > 0) {
            positions.reserve(
                static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));

            // Sample a regular grid over the unit bounds and let the spawn
            // operator decide whether each point is accepted.
            //
            // The flip flag inverts the acceptance result so the same spawn
            // operator can be reused for complement-style emission regions.
            for (int iz = 0; iz < nz; ++iz) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        const Vector3<T> sample(
                            lower.x + static_cast<T>(ix) * _spacing,
                            lower.y + static_cast<T>(iy) * _spacing,
                            lower.z + static_cast<T>(iz) * _spacing);

                        const bool accepted = spawn_op.spawn(geometry, sample, _tolerance);

                        if (_flip ? !accepted : accepted) {
                            positions.push_back(sample);
                        }
                    }
                }
            }
        }

        total_count += positions.size();

        // Upload the accepted local-space emission positions for this unit.
        _local_positions[i] = DeviceBuffer<Vector3<T>>(positions.begin(), positions.end());
    }

    // Reset species-related caches before rebuilding them.
    _species_cache.clear();
    _shuffled_species.clear();
    _shuffle_keys.clear();
    _local_particle_count = total_count;

    if (total_count == 0) {

        // Nothing will be emitted, so keep the cache valid but empty.
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    _species_cache.resize(total_count);
    _shuffled_species.resize(total_count);
    _shuffle_keys.resize(total_count);

    const int count         = static_cast<int>(total_count);
    const int species_count = static_cast<int>(generators.size());
    auto* cache             = atlas::raw_pointer_cast(this->_species_cache.data());

    // Build a deterministic base species sequence by cycling through all
    // available generator/species indices.
    //
    // This ensures that every cached spawn slot has an initial species
    // assignment before the sequence is shuffled later.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_DEVICE(const int i) {
            cache[i] = static_cast<std::size_t>(i % species_count);
        });

    _shuffle_seed         = 0;
    _is_invalidated_cache = false;
}

template <typename T>
void
Source<T>::shuffle_species(const std::size_t count) {

    // Keep the shuffled species buffer empty when there is nothing to emit.
    if (count == 0) {
        _shuffled_species.clear();
        _shuffle_keys.clear();
        return;
    }

    // Start from the deterministic cached species sequence and permute it using
    // a per-emission shuffle key set.
    _shuffled_species = _species_cache;

    const std::uint64_t seed = _shuffle_seed++;

    auto* keys = atlas::raw_pointer_cast(this->_shuffle_keys.data());
    const ShuffleOperator shuffle {};

    // Generate one sortable pseudo-random key per particle slot.
    atlas::parallel_for<ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            keys[i] = shuffle(static_cast<int>(i), seed);
        });

    // Sort the species buffer by the generated shuffle keys to obtain a
    // reproducibly shuffled species order.
    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        _shuffle_keys.begin(),
        _shuffle_keys.end(),
        _shuffled_species.begin());
}

template <typename T>
void
Source<T>::emit() {

    const std::size_t step_index = _step_index++;
    auto* source_sensor_matrics = _observer ? _observer->sensor_matrics<atlas::SourceSensorMatrics>() : nullptr;
    HostBuffer<std::size_t> emitted_per_unit;

    if (source_sensor_matrics != nullptr) {
        emitted_per_unit = HostBuffer<std::size_t>(_units.size(), std::size_t { 0 });
    }

    const auto record_source_metrics = [&] {
        if (source_sensor_matrics == nullptr) {
            return;
        }

        for (std::size_t unit_index = 0; unit_index < emitted_per_unit.size(); ++unit_index) {
            source_sensor_matrics->record(step_index, unit_index, emitted_per_unit[unit_index]);
        }
    };

    if (!_fluid) {
        record_source_metrics();
        return;
    }

    // Ensure local emission positions and species caches are ready.
    rebuild_cache();

    if (_local_particle_count == 0) {
        record_source_metrics();
        return;
    }

    auto& positions_buf        = _fluid->template state<FluidPositionState<T>>()->data();
    auto& velocities_buf       = _fluid->template state<FluidVelocityState<T>>()->data();
    auto& species_buf          = _fluid->template state<FluidSpeciesState<T>>()->data();
    auto& active_buf           = _fluid->template state<FluidActiveState<T>>()->data();
    const auto& generators_buf = _fluid->generators();
    const auto& properties_buf = _fluid->particle_properties();

    if (generators_buf.empty() || properties_buf.empty()) {
        record_source_metrics();
        return;
    }

    const std::size_t current_particle_count = _fluid->particle_count();
    const std::size_t available_slots        = current_particle_count < _fluid->buffer_size()
               ? (_fluid->buffer_size() - current_particle_count)
               : std::size_t { 0 };

    const std::size_t emit_count = _local_particle_count < available_slots
        ? _local_particle_count
        : available_slots;

    if (emit_count == 0) {
        atlas::logger::warn() << "\n"
                              << "Source emission skipped: no available slots for "
                              << _local_particle_count
                              << " particles\n";
        record_source_metrics();
        return;
    }

    // Randomize species assignment order for this emission pass.
    shuffle_species(emit_count);

    const auto* units        = atlas::raw_pointer_cast(this->_units.data());
    const auto* generators   = atlas::raw_pointer_cast(generators_buf.data());
    const auto* properties   = atlas::raw_pointer_cast(properties_buf.data());
    const auto* species      = atlas::raw_pointer_cast(this->_shuffled_species.data());
    auto* positions_out      = atlas::raw_pointer_cast(positions_buf.data());
    auto* velocities_out     = atlas::raw_pointer_cast(velocities_buf.data());
    auto* species_out        = atlas::raw_pointer_cast(species_buf.data());
    auto* active_out         = atlas::raw_pointer_cast(active_buf.data());
    const T temperature      = _temperature;
    const int property_count = static_cast<int>(properties_buf.size());
    const ShuffleOperator shuffle {};
    const std::uint64_t emission_seed = _shuffle_seed;

    std::size_t species_offset = 0;
    std::size_t emitted_count  = 0;
    int dst_offset             = static_cast<int>(current_particle_count);

    // Emit cached local positions unit by unit.
    //
    // For each accepted local-space sample:
    // 1. transform it into world space
    // 2. generate an initial velocity from the assigned species generator
    // 3. write species id and active flag
    for (std::size_t unit_index = 0; unit_index < _local_positions.size(); ++unit_index) {

        const auto& positions       = _local_positions[unit_index];
        const std::size_t remaining = emit_count - emitted_count;

        if (remaining == 0) {
            break;
        }

        const int count = static_cast<int>(positions.size() < remaining ? positions.size() : remaining);

        if (count == 0) {
            continue;
        }

        if (source_sensor_matrics != nullptr) {
            emitted_per_unit[unit_index] = static_cast<std::size_t>(count);
        }

        const auto* local_positions = atlas::raw_pointer_cast(positions.data());
        const auto* local_species   = species + species_offset;

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            count,
            [=] ATLAS_DEVICE(const int i) {
                const int dst = dst_offset + i;

                const size_t sid = local_species[i];
                if (sid >= static_cast<std::size_t>(property_count)) {
                    return;
                }

                auto generator = generators[sid];
                generator.reseed(static_cast<unsigned int>(shuffle(dst, emission_seed)));

                Vector3<T> world_pos;
                units[unit_index].sync_operator().sync_to_world(local_positions[i], world_pos);

                positions_out[dst]  = world_pos;
                velocities_out[dst] = generator.generate(temperature, properties[sid].molecular_mass);
                species_out[dst]    = sid;
                active_out[dst]     = 1;
            });

        species_offset += static_cast<std::size_t>(count);
        emitted_count += static_cast<std::size_t>(count);
        dst_offset += count;
    }

    _fluid->set_particle_count(current_particle_count + emitted_count);
    record_source_metrics();
}

template <typename T>
Source<T>
Source<T>::Builder::build() {

    // Validate all structural and numeric builder parameters before constructing
    // the final source object.
    validate();

    return Source<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<SpawnType>(_spawn_types.begin(), _spawn_types.end()),
        DeviceBuffer<SpawnOperator<T>>(_spawn_operators.begin(), _spawn_operators.end()),
        _fluid,
        _flip,
        _spacing,
        _tolerance,
        _temperature,
        _observer);
}

template <typename T>
atlas::host_shared_ptr<Source<T>>
Source<T>::Builder::make_host_shared() {

    // Construct a value object first, then move it into shared host ownership.
    return atlas::make_host_shared<Source<T>>(build());
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {

    if (units.empty()) {
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    // Append source units rather than replacing them, allowing the builder to
    // accumulate multiple emission units incrementally.
    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

    // Store the target fluid that will receive emitted particles.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_observer(ObserverHostPtr observer) noexcept {

    _observer = std::move(observer);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_types(const HostBuffer<SpawnType>& spawn_types) {

    if (spawn_types.empty()) {
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    // Append spawn-type configuration entries.
    _spawn_types.insert(_spawn_types.end(), spawn_types.begin(), spawn_types.end());
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept {

    // Add a single spawn operator entry.
    _spawn_operators.push_back(spawn_operator);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {

    if (spawn_operators.empty()) {
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    // Append spawn operators rather than replacing them.
    _spawn_operators.insert(_spawn_operators.end(), spawn_operators.begin(), spawn_operators.end());
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_tolerance(const T tolerance) noexcept {

    // Store the geometric acceptance tolerance used by spawn operators.
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_flip(const bool flip) noexcept {

    // Invert spawn acceptance when enabled.
    _flip = flip;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spacing(const T spacing) noexcept {

    // Store the regular sampling grid spacing used when building emission positions.
    _spacing = spacing;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_temperature(const T temperature) noexcept {

    // Store the temperature parameter forwarded to the velocity generators at emit time.
    _temperature = temperature;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {

    if (_units.empty()) {
        throw std::runtime_error("Source::Builder: units must not be empty.");
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

    // Spawn type configuration must be either:
    // - shared by all units with exactly one entry, or
    // - specified per unit with matching unit count.
    if (_spawn_types.size() != 1 && _spawn_types.size() != _units.size()) {
        throw std::runtime_error(
            "Source::Builder: spawn types must have size 1 or match unit count.");
    }

    // The same cardinality rule applies to spawn operators.
    if (_spawn_operators.size() != 1 && _spawn_operators.size() != _units.size()) {
        throw std::runtime_error(
            "Source::Builder: spawn operators must have size 1 or match unit count.");
    }

    if (!std::isfinite(_spacing) || _spacing <= T(0)) {
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    if (!std::isfinite(_tolerance)) {
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    if (!std::isfinite(_temperature)) {
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

}
