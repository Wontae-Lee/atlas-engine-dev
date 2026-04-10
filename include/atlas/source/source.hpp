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
Source<T>::Source(DeviceBuffer<Unit<T>> units,
                  DeviceBuffer<SpawnType> spawn_types,
                  DeviceBuffer<SpawnOperator<T>> spawn_operators,
                  FluidHostPtr<T> fluid,
                  const bool flip,
                  const T tolerance,
                  const T temperature) noexcept
    : _units(std::move(units))
    , _spawn_types(std::move(spawn_types))
    , _spawn_operators(std::move(spawn_operators))
    , _fluid(std::move(fluid))
    , _flip(flip)
    , _tolerance(tolerance)
    , _temperature(temperature)
    , _is_invalidated_cache(true) { }

template <typename T>
typename Source<T>::Builder
Source<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
Source<T>::update(const T dt) {
    if (_units.empty() || !(dt > T(0))) {
        return;
    }

    auto* units = atlas::raw_pointer_cast(_units.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        static_cast<int>(_units.size()),
        [units, dt] ATLAS_DEVICE(const int i) {
            units[i].update(dt);
        });
}

template <typename T>
void
Source<T>::rebuild_cache() noexcept {
    if (!_is_invalidated_cache) {
        return;
    }

    if (_units.empty() || _spawn_types.empty() || _spawn_operators.empty() || !_fluid || _fluid->empty()) {
        _local_positions.clear();
        _local_position_unit_indices.clear();
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    HostBuffer<Unit<T>> units_host(_units.size());
    HostBuffer<SpawnOperator<T>> spawn_operators_host(_spawn_operators.size());
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(_units.data()),
        units_host.data(),
        units_host.size());
    atlas::copy_device_to_host(
        atlas::raw_pointer_cast(_spawn_operators.data()),
        spawn_operators_host.data(),
        spawn_operators_host.size());

    HostBuffer<Vector3<T>> local_positions_host;
    HostBuffer<int> local_position_unit_indices_host;

    for (std::size_t i = 0; i < units_host.size(); ++i) {
        DeviceBuffer<Vector3<T>> local_positions_for_unit;
        const auto& unit             = units_host[i];
        const auto& geometry_op      = unit.geometry_operator();
        const std::size_t spawn_index = (spawn_operators_host.size() == 1 || i >= spawn_operators_host.size()) ? 0 : i;
        const auto spawn_operator    = spawn_operators_host[spawn_index];
        const bool flip              = _flip;

        atlas::sampling::sample_spawn_grid(
            local_positions_for_unit,
            geometry_op,
            _spacing,
            _tolerance,
            [spawn_operator, flip] ATLAS_DEVICE(const auto& query, const Vector3<T>& sample, T tol) {
                const bool should_spawn = spawn_operator.spawn(query, sample, tol);
                return flip ? !should_spawn : should_spawn;
            });

        if (local_positions_for_unit.empty()) {
            continue;
        }

        HostBuffer<Vector3<T>> local_positions_for_unit_host(local_positions_for_unit.size());
        atlas::copy_device_to_host(
            atlas::raw_pointer_cast(local_positions_for_unit.data()),
            local_positions_for_unit_host.data(),
            local_positions_for_unit_host.size());

        local_positions_host.insert(
            local_positions_host.end(),
            local_positions_for_unit_host.begin(),
            local_positions_for_unit_host.end());
        local_position_unit_indices_host.insert(
            local_position_unit_indices_host.end(),
            local_positions_for_unit_host.size(),
            static_cast<int>(i));
    }

    _local_positions = DeviceBuffer<Vector3<T>>(local_positions_host.begin(), local_positions_host.end());
    _local_position_unit_indices = DeviceBuffer<int>(
        local_position_unit_indices_host.begin(),
        local_position_unit_indices_host.end());

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
Source<T>::emit(FluidDeviceProbe<T>& particle_probe) {
    rebuild_cache();

    if (_units.empty() || !_fluid || _fluid->empty() || _local_positions.empty()) {
        return;
    }

    const int local_count = static_cast<int>(_local_positions.size());
    if (local_count <= 0) {
        return;
    }

    const int current_count  = particle_probe.particle_count;
    const int required_space = current_count + local_count;
    if (required_space > static_cast<int>(particle_probe.buffer_size)) {
        atlas::logger::warn()
            << "Source::emit: insufficient space. Required: " << required_space
            << ", Available: " << particle_probe.buffer_size;
        return;
    }

    const auto* units                 = atlas::raw_pointer_cast(_units.data());
    const auto* generators            = atlas::raw_pointer_cast(_fluid->generators().data());
    const auto* particle_properties   = atlas::raw_pointer_cast(_fluid->particles().data());
    const Vector3<T>* local_pos_ptr   = atlas::raw_pointer_cast(_local_positions.data());
    const int* local_unit_indices_ptr = atlas::raw_pointer_cast(_local_position_unit_indices.data());
    const T temperature               = _temperature;

    Vector3<T>* out_pos = particle_probe.pos + current_count;
    Vector3<T>* out_vel = particle_probe.vel + current_count;
    T* out_temperature  = particle_probe.temperature + current_count;
    size_t* out_species = particle_probe.species + current_count;

    _shuffled_species = _species_cache;

    const std::uint64_t seed        = _shuffle_seed++;
    std::uint64_t* shuffle_keys_ptr = atlas::raw_pointer_cast(_shuffle_keys.data());
    const ShuffleOperator shuffle_op {};

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        local_count,
        [=] ATLAS_DEVICE(const int i) {
            shuffle_keys_ptr[i] = shuffle_op(i, seed);
        });

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        _shuffle_keys.begin(),
        _shuffle_keys.end(),
        _shuffled_species.begin());

    const size_t* shuffled_species_ptr = atlas::raw_pointer_cast(_shuffled_species.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        local_count,
        [=] ATLAS_DEVICE(const int i) {
            const Vector3<T>& local_p = local_pos_ptr[i];
            const auto& sync_op       = units[local_unit_indices_ptr[i]].sync_operator();
            const size_t species_id   = shuffled_species_ptr[i];

            Vector3<T> world_p;
            sync_op.sync_to_world(local_p, world_p);
            out_pos[i] = world_p;

            out_vel[i]         = generators[species_id].generate(temperature, particle_properties[species_id].mass);
            out_temperature[i] = temperature;
            out_species[i]     = species_id;
        });

    particle_probe.particle_count += local_count;
}

template <typename T>
void
Source<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {
    _units                 = std::move(units);
    _is_invalidated_cache  = true;
}

template <typename T>
void
Source<T>::set_units(const HostBuffer<Unit<T>>& units) {
    if (units.empty()) {
        atlas::logger::error()
            << "Source: units must not be empty.";
        throw std::runtime_error("Source: units must not be empty.");
    }

    _units = DeviceBuffer<Unit<T>>(units.begin(), units.end());
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
Source<T>::set_spawn_types(DeviceBuffer<SpawnType> spawn_types) noexcept {
    _spawn_types          = std::move(spawn_types);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_types(const HostBuffer<SpawnType>& spawn_types) {
    if (spawn_types.empty()) {
        atlas::logger::error()
            << "Source: spawn types must not be empty.";
        throw std::runtime_error("Source: spawn types must not be empty.");
    }

    _spawn_types = DeviceBuffer<SpawnType>(spawn_types.begin(), spawn_types.end());
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_operators(DeviceBuffer<SpawnOperator<T>> spawn_operators) noexcept {
    _spawn_operators      = std::move(spawn_operators);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {
    if (spawn_operators.empty()) {
        atlas::logger::error()
            << "Source: spawn operators must not be empty.";
        throw std::runtime_error("Source: spawn operators must not be empty.");
    }

    _spawn_operators = DeviceBuffer<SpawnOperator<T>>(spawn_operators.begin(), spawn_operators.end());
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
Source<T>::set_temperature(const T temperature) noexcept {
    _temperature = temperature;
}

template <typename T>
DeviceBuffer<Unit<T>>&
Source<T>::units() noexcept {
    return _units;
}

template <typename T>
const DeviceBuffer<Unit<T>>&
Source<T>::units() const noexcept {
    return _units;
}

template <typename T>
const FluidHostPtr<T>&
Source<T>::fluid() const noexcept {
    return _fluid;
}

template <typename T>
DeviceBuffer<SpawnType>&
Source<T>::spawn_types() noexcept {
    return _spawn_types;
}

template <typename T>
const DeviceBuffer<SpawnType>&
Source<T>::spawn_types() const noexcept {
    return _spawn_types;
}

template <typename T>
DeviceBuffer<SpawnOperator<T>>&
Source<T>::spawn_operators() noexcept {
    return _spawn_operators;
}

template <typename T>
const DeviceBuffer<SpawnOperator<T>>&
Source<T>::spawn_operators() const noexcept {
    return _spawn_operators;
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
T
Source<T>::temperature() const noexcept {
    return _temperature;
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

    Source<T> source(
        DeviceBuffer<Unit<T>>(_units.begin(), _units.end()),
        DeviceBuffer<SpawnType>(_spawn_types.begin(), _spawn_types.end()),
        DeviceBuffer<SpawnOperator<T>>(_spawn_operators.begin(), _spawn_operators.end()),
        _fluid,
        _flip,
        _tolerance,
        _temperature);
    source._spacing = _spacing;
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
Source<T>::Builder::with_units(const HostBuffer<Unit<T>>& units) {
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
    _fluid = std::move(fluid);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_types(const HostBuffer<SpawnType>& spawn_types) {
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
    _spawn_operators.push_back(spawn_operator);
    return *this;
}

template <typename T>
typename Source<T>::Builder&
Source<T>::Builder::with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators) {
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
Source<T>::Builder::with_temperature(const T temperature) noexcept {
    _temperature = temperature;
    return *this;
}

template <typename T>
void
Source<T>::Builder::validate() const {
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
        atlas::logger::error()
            << "Source::Builder: spawn types must have size 1 or match unit count.";
        throw std::runtime_error(
            "Source::Builder: spawn types must have size 1 or match unit count.");
    }

    if (_spawn_operators.size() != 1 && _spawn_operators.size() != _units.size()) {
        atlas::logger::error()
            << "Source::Builder: spawn operators must have size 1 or match unit count.";
        throw std::runtime_error(
            "Source::Builder: spawn operators must have size 1 or match unit count.");
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

    if (!std::isfinite(_temperature)) {
        atlas::logger::error()
            << "Source::Builder: temperature must be finite.";
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }

}

}
