#pragma once

#include <atlas/iterator/zip_iterator.h>
#include <atlas/logging/logging.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>
#include <atlas/scan/exclusive_scan.h>
#include <atlas/tuple/tuple.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Source<T>::Source(Unit<T> unit,
                  FluidHostPtr<T> fluid,
                  const SpawnType spawn_type,
                  const bool flip,
                  const T tolerance) noexcept
    : _unit(std::move(unit))
    , _fluid(std::move(fluid))
    , _spawn_operator(spawn_type)
    , _flip(flip)
    , _tolerance(tolerance)
    , _is_invalidated_cache(true) { }

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Source<T>::rebuild_cache() noexcept {
    // Cache rebuilds are intentionally coarse-grained: any structural change
    // to unit/fluid/spawn parameters flips `_is_invalidated_cache`, and the next emit
    // reconstructs both spawnable positions and the baseline species layout.
    if (!_is_invalidated_cache) {
        return;
    }

    if (!_fluid || _fluid->empty()) {
        // An empty fluid means the source is effectively disabled; clear every
        // cache so later emits become cheap early-outs without stale data.
        _local_positions.clear();
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    const auto geometry_op = _unit.geometry_operator();
    const bool flip        = _flip;
    atlas::sampling::sample_spawn_grid(
        _local_positions,
        geometry_op,
        _spacing,
        _tolerance,
        // sample_spawn_grid() already performs predicate filtering and compaction,
        // so `_local_positions` contains only spawn-eligible samples afterwards.
        [spawn_op = _spawn_operator, flip] ATLAS_DEVICE(const auto& query, const Vector3<T>& sample, T tol) {
            const bool should_spawn = spawn_op.spawn(query, sample, tol);
            return flip ? !should_spawn : should_spawn;
        });

    const int local_count = static_cast<int>(_local_positions.size());
    _species_cache.clear();
    _shuffled_species.clear();
    _shuffle_keys.clear();

    if (local_count > 0) {
        const auto& mole_fractions = _fluid->mole_fractions();
        const int num_species      = static_cast<int>(mole_fractions.size());
        HostBuffer<size_t> species_cache_host(static_cast<std::size_t>(local_count));

        _shuffled_species.resize(static_cast<std::size_t>(local_count));
        _species_cache.resize(static_cast<std::size_t>(local_count));
        _shuffle_keys.resize(static_cast<std::size_t>(local_count));

        for (int i = 0; i < local_count; ++i) {
            // Build a deterministic baseline species layout from the cumulative
            // mole-fraction distribution. Later emits only reshuffle this cache.
            const T fraction   = static_cast<T>(i) / static_cast<T>(local_count);
            size_t species_idx = 0;
            T running_sum      = T(0);

            for (int s = 0; s < num_species; ++s) {
                running_sum += mole_fractions[s];
                if (fraction < running_sum) {
                    species_idx = static_cast<size_t>(s);
                    break;
                }
            }

            species_cache_host[static_cast<std::size_t>(i)] = species_idx;
        }

        atlas::copy_host_to_device(
            species_cache_host.data(),
            atlas::raw_pointer_cast(_species_cache.data()),
            static_cast<std::size_t>(local_count));
    }

    _shuffle_seed         = 0;
    _is_invalidated_cache = false;
}

template <typename T>
void
Source<T>::emit(ParticleDeviceProbe<T>& particle_probe) {
    // Rebuild-on-demand keeps setter APIs cheap and pushes the heavy work to
    // the first emit that actually needs fresh cached state.
    rebuild_cache();

    if (!_fluid || _fluid->empty() || _local_positions.empty()) {
        return;
    }

    const int local_count           = static_cast<int>(_local_positions.size());
    const Vector3<T>* local_pos_ptr = atlas::raw_pointer_cast(_local_positions.data());
    const int spawn_count           = local_count;
    if (spawn_count <= 0) {
        return;
    }

    // Emission appends into the active prefix, so the source must reject writes
    // that would overflow the probe's preallocated storage.
    const int current_count  = particle_probe.particle_count;
    const int required_space = current_count + spawn_count;
    if (required_space > static_cast<int>(particle_probe.buffer_size)) {
        atlas::logger::warn()
            << "Source::emit: insufficient space. Required: " << required_space
            << ", Available: " << particle_probe.buffer_size;
        return;
    }

    // Prepare append pointers into the active suffix that this emit call owns.
    const auto sync_op = _unit.sync_operator();
    const auto gen_op  = _generate_operator;

    Vector3<T>* out_pos = particle_probe.pos + current_count;
    Vector3<T>* out_vel = particle_probe.vel + current_count;
    size_t* out_species = particle_probe.species + current_count;

    // Start from the deterministic baseline species arrangement and permute it
    // via random sort keys generated on the active backend buffer.
    _shuffled_species = _species_cache;

    const std::uint64_t seed        = _shuffle_seed++;
    std::uint64_t* shuffle_keys_ptr = atlas::raw_pointer_cast(_shuffle_keys.data());
    const ShuffleOperator shuffle_op {};

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        local_count,
        [=](const int i) {
            shuffle_keys_ptr[i] = shuffle_op(i, seed);
        });

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        _shuffle_keys.begin(),
        _shuffle_keys.end(),
        _shuffled_species.begin());

    const size_t* shuffled_species_ptr = atlas::raw_pointer_cast(_shuffled_species.data());

    // The current Source API still uses the runtime GenerateOperator directly.
    // Species-specific generator parameters can be threaded in here later.
    const T param0 = T(0);
    const T param1 = T(1);

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        local_count,
        [=](const int i) {
            const Vector3<T>& local_p = local_pos_ptr[i];

            // Transform accepted local samples into world-space particle state.
            Vector3<T> world_p;
            sync_op.sync_to_world(local_p, world_p);
            out_pos[i] = world_p;

            // Each emitted slot receives a freshly generated velocity and a
            // species tag from the shuffled cache computed above.
            out_vel[i]     = gen_op.generate(param0, param1);
            out_species[i] = shuffled_species_ptr[i];
        });

    particle_probe.particle_count += spawn_count;
}

template <typename T>
void
Source<T>::set_unit(Unit<T> unit) noexcept {
    _unit                 = std::move(unit);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid                = std::move(fluid);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_type(const SpawnType spawn_type) noexcept {
    _spawn_operator       = SpawnOperator<T>(spawn_type);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_operator(const SpawnOperator<T> spawn_operator) noexcept {
    _spawn_operator       = spawn_operator;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_tolerance(const T tolerance) noexcept {
    _tolerance            = tolerance;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_flip(const bool flip) noexcept {
    _flip                 = flip;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spacing(const T spacing) noexcept {
    _spacing              = spacing;
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_generate_operator(const GenerateOperator<T> generate_operator) noexcept {
    _generate_operator = generate_operator;
}

template <typename T>
const Unit<T>&
Source<T>::unit() const noexcept {
    return _unit;
}

template <typename T>
const FluidHostPtr<T>&
Source<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
SpawnType
Source<T>::spawn_type() const noexcept {
    return _spawn_operator.type;
}

template <typename T>
const SpawnOperator<T>&
Source<T>::spawn_operator() const noexcept {
    return _spawn_operator;
}

template <typename T>
T
Source<T>::tolerance() const noexcept {
    return _tolerance;
}

template <typename T>
bool
Source<T>::flip() const noexcept {
    return _flip;
}

template <typename T>
T
Source<T>::spacing() const noexcept {
    return _spacing;
}

template <typename T>
const GenerateOperator<T>&
Source<T>::generate_operator() const noexcept {
    return _generate_operator;
}

template <typename T>
const DeviceBuffer<Vector3<T>>&
Source<T>::local_positions() const noexcept {
    return _local_positions;
}

template <typename T>
Source<T>
Source<T>::Builder::build() {
    validate();
    Source<T> source(std::move(*_unit), _fluid, _spawn_type, _flip, _tolerance);
    source._spacing              = _spacing;
    source._generate_operator    = _generate_operator;
    source._is_invalidated_cache = true;
    return source;
}

template <typename T>
atlas::host_shared_ptr<Source<T>>
Source<T>::Builder::make_host_shared() {
    return atlas::make_host_shared<Source<T>>(build());
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_unit(const Unit<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_unit(Unit<T>&& unit) noexcept {
    _unit = std::move(unit);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_fluid(FluidHostPtr<T> fluid) noexcept {
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_type(const SpawnType spawn_type) noexcept {
    _spawn_type = spawn_type;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_tolerance(const T tolerance) noexcept {
    _tolerance = tolerance;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_flip(const bool flip) noexcept {
    _flip = flip;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spacing(const T spacing) noexcept {
    _spacing = spacing;
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_generate_operator(const GenerateOperator<T> generate_operator) noexcept {
    _generate_operator = generate_operator;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {
    if (!_unit.has_value()) {
        atlas::logger::error()
            << "Source::Builder: unit must be provided.";
        throw std::runtime_error("Source::Builder: unit must be provided.");
    }

    if (!_fluid) {
        atlas::logger::error()
            << "Source::Builder: fluid must be provided.";
        throw std::runtime_error("Source::Builder: fluid must be provided.");
    }

    if (!std::isfinite(_tolerance)) {
        atlas::logger::error()
            << "Source::Builder: tolerance must be finite.";
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    if (!std::isfinite(_spacing) || _spacing <= T(0)) {
        atlas::logger::error()
            << "Source::Builder: spacing must be finite and positive.";
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }
}

} // namespace atlas::system
