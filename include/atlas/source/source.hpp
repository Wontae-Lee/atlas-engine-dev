#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Source<T>::Source(DeviceBuffer<Unit<T>> units,
                  DeviceBuffer<SpawnType> spawn_types,
                  DeviceBuffer<SpawnOperator<T>> spawn_operators,
                  FluidHostPtr<T> fluid,
                  const bool flip,
                  const T spacing,
                  const T tolerance,
                  const T temperature) noexcept
    : _units(std::move(units))
    , _spawn_types(std::move(spawn_types))
    , _spawn_operators(std::move(spawn_operators))
    , _fluid(std::move(fluid))
    , _flip(flip)
    , _spacing(spacing)
    , _tolerance(tolerance)
    , _temperature(temperature)
    , _is_invalidated_cache(true) {
    // Construct a source from already-materialized device buffers and runtime parameters.
    //
    // Stored state:
    // - _units            : emitter units that define source geometry / transforms
    // - _spawn_types      : per-unit or broadcast spawn classification metadata
    // - _spawn_operators  : per-unit or broadcast spawn predicates
    // - _fluid            : fluid description used to assign species / generators
    // - _flip             : invert the acceptance test of the spawn predicate if true
    // - _spacing          : regular grid spacing used to sample candidate spawn points
    // - _tolerance        : tolerance forwarded into spawn predicate evaluation
    // - _temperature      : emitted particle temperature
    //
    // Cache policy:
    // - mark the cache invalid initially so the first emit() call rebuilds all
    //   derived spawn-position/species data.
}

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {
    // Return a fresh builder for staged Source<T> construction.
    return Builder {};
}

template <typename T>
void
Source<T>::update(const T dt) {
    // Advance every source unit using its own kinematic state.
    //
    // This keeps moving / rotating emitters synchronized with simulation time
    // before particle emission is evaluated.

    // Nothing to update if there are no units or dt is non-positive.
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());
    // Raw pointer to device-side unit storage for backend execution.

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            // Advance one unit by dt.
            units[i].update(dt);
        });
}

template <typename T>
void
Source<T>::rebuild_cache() noexcept {
    // Rebuild all cached spawn data derived from:
    // - unit geometry
    // - spawn operators
    // - fluid mole fractions
    // - source spacing / tolerance / flip mode
    //
    // Cached outputs:
    // - _local_positions  : accepted local-space spawn positions per unit
    // - _species_cache    : deterministic per-particle species assignment before shuffling
    // - _shuffled_species : shuffled species order used at emission time
    // - _shuffle_keys     : temporary key buffer used for species shuffling
    //
    // Rebuild is skipped unless some setter or constructor marked the cache invalid.
    if (!_is_invalidated_cache) {
        return;
    }

    // If required inputs are missing, clear all caches and mark them valid-empty.
    if (_units.empty() || _spawn_types.empty() || _spawn_operators.empty() || !_fluid || _fluid->empty()) {
        _local_positions.clear();
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    const auto& mole_fractions = _fluid->mole_fractions();
    // Species mixture weights used to construct the deterministic species cache.

    // If the fluid has no mixture information, emission cannot assign species.
    if (mole_fractions.empty()) {
        _local_positions.clear();
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    // Pull the current units and spawn operators to host so cache generation can
    // evaluate regular sampling loops and predicate logic on the host side.
    const HostBuffer<Unit<T>> units(_units.begin(), _units.end());
    const HostBuffer<SpawnOperator<T>> spawn_operators(
        _spawn_operators.begin(),
        _spawn_operators.end());

    _local_positions.clear();
    _local_positions.resize(units.size());
    // Prepare one local-position buffer per unit.

    std::size_t total_count                = 0;
    const std::size_t spawn_operator_count = spawn_operators.size();
    // Total number of accepted spawn positions across all units.
    // Spawn operators may be:
    // - broadcast: count == 1
    // - per-unit : count == units.size()

    for (std::size_t i = 0; i < units.size(); ++i) {
        const auto& geometry = units[i].geometry_operator();
        // Geometry query operator of the i-th unit.

        const auto bounds = geometry.bound();
        // Axis-aligned bounding box used to build a regular candidate grid.

        const auto& lower = bounds.lower_corner;
        const auto& upper = bounds.upper_corner;

        const auto& spawn_op = spawn_operators[(spawn_operator_count == 1) ? 0 : i];
        // Support either one broadcast operator or one operator per unit.

        const int nx = atlas::sampling::sample_axis_count(lower.x, upper.x, _spacing);
        const int ny = atlas::sampling::sample_axis_count(lower.y, upper.y, _spacing);
        const int nz = atlas::sampling::sample_axis_count(lower.z, upper.z, _spacing);
        // Number of regular samples along each axis of the unit bounding box.

        HostBuffer<Vector3<T>> positions;
        // Host-side temporary list of accepted local-space spawn positions for unit i.

        if (nx > 0 && ny > 0 && nz > 0) {
            positions.reserve(
                static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));
            // Reserve the full candidate-grid capacity to reduce reallocations.

            for (int iz = 0; iz < nz; ++iz) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        const Vector3<T> sample(
                            lower.x + static_cast<T>(ix) * _spacing,
                            lower.y + static_cast<T>(iy) * _spacing,
                            lower.z + static_cast<T>(iz) * _spacing);
                        // Regular local-space sample point inside the AABB grid.

                        const bool accepted = spawn_op.spawn(geometry, sample, _tolerance);
                        // Query whether this sample point is accepted by the spawn predicate.

                        // If _flip is false:
                        // - keep accepted samples
                        //
                        // If _flip is true:
                        // - keep rejected samples instead
                        if (_flip ? !accepted : accepted) {
                            positions.push_back(sample);
                        }
                    }
                }
            }
        }

        total_count += positions.size();
        // Accumulate the total number of cached spawn points across all units.

        _local_positions[i] = DeviceBuffer<Vector3<T>>(positions.begin(), positions.end());
        // Move this unit's accepted local-space positions into device storage.
    }

    _species_cache.clear();
    _shuffled_species.clear();
    _shuffle_keys.clear();
    // Drop any previous species/shuffle data before regenerating it.

    if (total_count == 0) {
        // Valid cache state with no accepted spawn samples.
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    _species_cache.resize(total_count);
    _shuffled_species.resize(total_count);
    _shuffle_keys.resize(total_count);
    // Prepare one species slot and one shuffle key per cached spawn point.

    const int count         = static_cast<int>(total_count);
    const int species_count = static_cast<int>(mole_fractions.size());
    const auto* fractions   = atlas::raw_pointer_cast(mole_fractions.data());
    auto* cache             = atlas::raw_pointer_cast(this->_species_cache.data());
    // Raw device pointers used to populate deterministic species assignments.

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_DEVICE(const int i) {
            const T fraction = static_cast<T>(i) / static_cast<T>(count);
            // Map the global spawn-point index into [0, 1) so species can be
            // assigned according to cumulative mole fractions.

            T sum          = T(0);
            size_t species = 0;
            // Default to species 0 unless a later cumulative threshold is reached.

            for (int s = 0; s < species_count; ++s) {
                sum += fractions[s];
                if (fraction < sum) {
                    species = static_cast<size_t>(s);
                    break;
                }
            }

            cache[i] = species;
            // Store the deterministic species id before shuffling.
        });

    _shuffle_seed         = 0;
    _is_invalidated_cache = false;
    // The cache is now fully rebuilt and coherent with the current source state.
}

template <typename T>
void
Source<T>::shuffle_species(const std::size_t count) {
    // Shuffle the deterministic species cache into a new randomized ordering.
    //
    // Purpose:
    // - preserve the global species counts implied by the deterministic cache
    // - vary local species ordering between emission calls
    //
    // Mechanism:
    // - copy _species_cache into _shuffled_species
    // - generate one 64-bit shuffle key per slot
    // - sort species by those keys

    if (count == 0) {
        _shuffled_species.clear();
        _shuffle_keys.clear();
        return;
    }

    _shuffled_species = _species_cache;
    // Start from the deterministic species assignment.

    const std::uint64_t seed = _shuffle_seed++;
    // Use and then advance the shuffle seed so subsequent emissions reshuffle differently.

    auto* keys = atlas::raw_pointer_cast(this->_shuffle_keys.data());
    const ShuffleOperator shuffle {};
    // Stateless key generator used to derive sortable pseudo-random keys.

    atlas::parallel_for<ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            keys[i] = shuffle(static_cast<int>(i), seed);
            // Generate one shuffle key per species slot.
        });

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        _shuffle_keys.begin(),
        _shuffle_keys.end(),
        _shuffled_species.begin());
    // Permute species ids according to the generated random-like key order.
}

template <typename T>
void
Source<T>::emit(FluidDeviceProbe<T>& particle_probe) {
    // Ensure all cached source-side data is up to date before emission.
    // rebuild_cache() prepares local spawn positions and species caches.
    rebuild_cache();

    // Abort emission if required source state is missing.
    // (no units, no fluid, or no cached spawn positions)
    if (_units.empty() || !_fluid || _fluid->empty() || _local_positions.empty()) {
        return;
    }

    // Access per-species generators and material properties from the fluid.
    const auto& generators_buf = _fluid->generators();
    const auto& particles_buf  = _fluid->particles();

    // Cannot emit if generator or particle data is missing.
    if (generators_buf.empty() || particles_buf.empty()) {
        return;
    }

    // Count total number of particles to be emitted across all units.
    std::size_t total_count = 0;
    for (const auto& positions : _local_positions) {
        total_count += positions.size();
    }

    // Nothing to emit if no accepted spawn positions exist.
    if (total_count == 0) {
        return;
    }

    // Determine destination write range in the particle buffer.
    const int begin = particle_probe.particle_count;
    const int end   = begin + static_cast<int>(total_count);

    // Abort if destination buffer does not have enough capacity.
    if (end > static_cast<int>(particle_probe.buffer_size)) {
        atlas::logger::warn()
            << "Source::emit: insufficient space. Required: " << end
            << ", Available: " << particle_probe.buffer_size;
        return;
    }

    // Randomize species ordering for this emission event.
    // _species_cache is deterministic; shuffling removes spatial bias.
    shuffle_species(total_count);

    // Extract raw pointers for device-side execution.
    const auto* units      = atlas::raw_pointer_cast(_units.data());
    const auto* generators = atlas::raw_pointer_cast(generators_buf.data());
    const auto* particles  = atlas::raw_pointer_cast(particles_buf.data());
    const auto* species    = atlas::raw_pointer_cast(_shuffled_species.data());
    const T temperature    = _temperature;

    // Track offsets into species array and destination buffer.
    std::size_t species_offset = 0;
    int dst_offset             = begin;

    // Iterate over each source unit and emit its particles.
    for (std::size_t unit_index = 0; unit_index < _local_positions.size(); ++unit_index) {

        const auto& positions = _local_positions[unit_index];
        const int count       = static_cast<int>(positions.size());

        // Skip units with no spawn positions.
        if (count == 0) {
            continue;
        }

        // Get local positions and corresponding species slice.
        const auto* local_positions = atlas::raw_pointer_cast(positions.data());
        const auto* local_species   = species + species_offset;

        // Emit particles in parallel.
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            count,
            [=] ATLAS_DEVICE(const int i) {
                // Compute destination index for this particle.
                const int dst = dst_offset + i;

                // Species id assigned after shuffling.
                const size_t sid = local_species[i];

                // Transform local-space position to world space.
                Vector3<T> world_pos;
                units[unit_index].sync_operator().sync_to_world(local_positions[i], world_pos);

                // Write particle position.
                particle_probe.pos[dst] = world_pos;

                // Generate and write particle velocity.
                particle_probe.vel[dst] = generators[sid].generate(temperature, particles[sid].mass);

                // Write particle temperature.
                particle_probe.temperature[dst] = temperature;

                // Write particle species id.
                particle_probe.species[dst] = sid;
            });

        // Advance species offset for next unit.
        species_offset += positions.size();

        // Advance destination offset for next unit.
        dst_offset += count;
    }

    // Update total particle count after emission completes.
    particle_probe.particle_count = end;

    // Log emitted particle count.
    atlas::logger::info() << "\n"
                          << "Source::emit: emitted " << total_count << " particles.";
}
template <typename T>
void
Source<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {
    // Replace device-side emitter units.
    //
    // Any change to unit geometry or transforms invalidates cached spawn positions.
    _units                = std::move(units);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_units(const HostBuffer<Unit<T>>& units) {
    // Replace emitter units from host-side storage.
    //
    // Empty unit lists are rejected because a source without units cannot emit.

    if (units.empty()) {
        atlas::logger::error()
            << "Source: units must not be empty.";
        throw std::runtime_error("Source: units must not be empty.");
    }

    _units                = DeviceBuffer<Unit<T>>(units.begin(), units.end());
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_fluid(FluidHostPtr<T> fluid) noexcept {
    // Replace the fluid description used for:
    // - species mixture fractions
    // - per-species material properties
    // - per-species velocity generators
    _fluid                = std::move(fluid);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_types(DeviceBuffer<SpawnType> spawn_types) noexcept {
    // Replace spawn-type metadata.
    //
    // Even if not directly consulted inside rebuild_cache(), this is still treated
    // as source-configuration state and therefore invalidates the cache.
    _spawn_types          = std::move(spawn_types);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_types(const HostBuffer<SpawnType>& spawn_types) {
    // Replace spawn-type metadata from host-side storage.

    if (spawn_types.empty()) {
        atlas::logger::error()
            << "Source: spawn types must not be empty.";
        throw std::runtime_error("Source: spawn types must not be empty.");
    }

    _spawn_types          = DeviceBuffer<SpawnType>(spawn_types.begin(), spawn_types.end());
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_operators(DeviceBuffer<SpawnOperator<T>> spawn_operators) noexcept {
    // Replace spawn operators.
    //
    // This directly changes which regular grid samples are accepted.
    _spawn_operators      = std::move(spawn_operators);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {
    // Replace spawn operators from host-side storage.

    if (spawn_operators.empty()) {
        atlas::logger::error()
            << "Source: spawn operators must not be empty.";
        throw std::runtime_error("Source: spawn operators must not be empty.");
    }

    _spawn_operators      = DeviceBuffer<SpawnOperator<T>>(spawn_operators.begin(), spawn_operators.end());
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_tolerance(const T tolerance) noexcept {
    // Update predicate tolerance.
    //
    // This may change which samples pass the spawn test.
    _tolerance            = tolerance;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_flip(const bool flip) noexcept {
    // Toggle inversion of spawn acceptance.
    //
    // false -> keep accepted points
    // true  -> keep rejected points
    _flip                 = flip;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spacing(const T spacing) noexcept {
    // Update regular grid spacing used to sample candidate spawn positions.
    //
    // This changes both candidate count and accepted positions.
    _spacing              = spacing;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_temperature(const T temperature) noexcept {
    // Update emission temperature.
    //
    // This does not affect cached spawn positions/species layout, so the cache
    // remains valid.
    _temperature = temperature;
}

template <typename T>
DeviceBuffer<Unit<T>>&
Source<T>::units() noexcept {
    // Mutable access to emitter units.
    return _units;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
Source<T>::units() const noexcept {
    // Const access to emitter units.
    return _units;
}

template <typename T>
const FluidHostPtr<T>&
Source<T>::fluid() const noexcept {
    // Const access to the bound fluid description.
    return _fluid;
}

template <typename T>
DeviceBuffer<SpawnType>&
Source<T>::spawn_types() noexcept {
    // Mutable access to spawn-type metadata.
    return _spawn_types;
}

template <typename T>
const DeviceBuffer<SpawnType>&
Source<T>::spawn_types() const noexcept {
    // Const access to spawn-type metadata.
    return _spawn_types;
}

template <typename T>
DeviceBuffer<SpawnOperator<T>>&
Source<T>::spawn_operators() noexcept {
    // Mutable access to spawn operators.
    return _spawn_operators;
}

template <typename T>
const DeviceBuffer<SpawnOperator<T>>&
Source<T>::spawn_operators() const noexcept {
    // Const access to spawn operators.
    return _spawn_operators;
}

template <typename T>
T
Source<T>::tolerance() const noexcept {
    // Return the current spawn predicate tolerance.
    return _tolerance;
}

template <typename T>
bool
Source<T>::flip() const noexcept {
    // Return whether spawn acceptance is inverted.
    return _flip;
}

template <typename T>
T
Source<T>::spacing() const noexcept {
    // Return the current regular sampling spacing.
    return _spacing;
}

template <typename T>
T
Source<T>::temperature() const noexcept {
    // Return the current emission temperature.
    return _temperature;
}

template <typename T>
const HostBuffer<DeviceBuffer<Vector3<T>>>&
Source<T>::local_positions() const noexcept {
    // Return cached accepted local-space spawn positions for all units.
    //
    // Layout:
    // - one DeviceBuffer<Vector3<T>> per unit
    return _local_positions;
}

template <typename T>
Source<T>
Source<T>::Builder::build() {
    // Validate builder configuration before constructing the final Source<T>.
    validate();

    return Source<T>(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<SpawnType>(_spawn_types.begin(), _spawn_types.end()),
        DeviceBuffer<SpawnOperator<T>>(_spawn_operators.begin(), _spawn_operators.end()),
        _fluid,
        _flip,
        _spacing,
        _tolerance,
        _temperature);
    // Materialize device buffers from staged host-side builder arrays and pass
    // through the remaining scalar / shared runtime configuration.
}

template <typename T>
atlas::host_shared_ptr<Source<T>>
Source<T>::Builder::make_host_shared() {
    // Build a Source<T> by value and then move it into host-shared storage.
    return atlas::make_host_shared<Source<T>>(build());
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
    // Append emitter units to the builder.

    if (units.empty()) {
        atlas::logger::error()
            << "Source::Builder: units must not be empty.";
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    _units.insert(_units.end(), units.begin(), units.end());
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    // Bind the fluid description used by the source.
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_types(const HostBuffer<SpawnType>& spawn_types) {
    // Append spawn-type metadata entries to the builder.

    if (spawn_types.empty()) {
        atlas::logger::error()
            << "Source::Builder: spawn types must not be empty.";
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    _spawn_types.insert(_spawn_types.end(), spawn_types.begin(), spawn_types.end());
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept {
    // Append a single spawn operator to the builder.
    _spawn_operators.push_back(spawn_operator);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {
    // Append multiple spawn operators to the builder.

    if (spawn_operators.empty()) {
        atlas::logger::error()
            << "Source::Builder: spawn operators must not be empty.";
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    _spawn_operators.insert(_spawn_operators.end(), spawn_operators.begin(), spawn_operators.end());
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_tolerance(const T tolerance) noexcept {
    // Stage spawn predicate tolerance.
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_flip(const bool flip) noexcept {
    // Stage spawn acceptance inversion mode.
    _flip = flip;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spacing(const T spacing) noexcept {
    // Stage regular sampling spacing.
    _spacing = spacing;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_temperature(const T temperature) noexcept {
    // Stage emission temperature.
    _temperature = temperature;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {
    // Validate all staged builder inputs before build() materializes device buffers.

    if (_units.empty()) {
        atlas::logger::error()
            << "Source::Builder: units must not be empty.";
        throw std::runtime_error("Source::Builder: units must not be empty.");
    }

    if (!_fluid) {
        atlas::logger::error()
            << "Source::Builder: fluid must be provided.";
        throw std::runtime_error("Source::Builder: fluid must be provided.");
    }

    if (_spawn_types.empty()) {
        atlas::logger::error()
            << "Source::Builder: spawn types must not be empty.";
        throw std::runtime_error("Source::Builder: spawn types must not be empty.");
    }

    if (_spawn_operators.empty()) {
        atlas::logger::error()
            << "Source::Builder: spawn operators must not be empty.";
        throw std::runtime_error("Source::Builder: spawn operators must not be empty.");
    }

    if (_spawn_types.size() != 1 && _spawn_types.size() != _units.size()) {
        // Spawn types must either:
        // - broadcast to all units with exactly one entry, or
        // - provide one entry per unit
        atlas::logger::error()
            << "Source::Builder: spawn types must have size 1 or match unit count.";
        throw std::runtime_error(
            "Source::Builder: spawn types must have size 1 or match unit count.");
    }

    if (_spawn_operators.size() != 1 && _spawn_operators.size() != _units.size()) {
        // Spawn operators must either:
        // - broadcast to all units with exactly one entry, or
        // - provide one entry per unit
        atlas::logger::error()
            << "Source::Builder: spawn operators must have size 1 or match unit count.";
        throw std::runtime_error(
            "Source::Builder: spawn operators must have size 1 or match unit count.");
    }

    if (!std::isfinite(_spacing) || _spacing <= T(0)) {
        // Regular grid spacing must define a valid positive finite interval.
        atlas::logger::error()
            << "Source::Builder: spacing must be finite and positive.";
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    if (!std::isfinite(_tolerance)) {
        // Tolerance must be finite so spawn predicate queries remain well-defined.
        atlas::logger::error()
            << "Source::Builder: tolerance must be finite.";
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    if (!std::isfinite(_temperature)) {
        // Temperature must be finite because it is written directly into particle
        // attributes and passed to generators.
        atlas::logger::error()
            << "Source::Builder: temperature must be finite.";
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

} // namespace atlas::system