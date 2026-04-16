#pragma once

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/parallel/parallel_sort.h>

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
}

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
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    const auto& mole_fractions = _fluid->mole_fractions();

    if (mole_fractions.empty()) {
        _local_positions.clear();
        _species_cache.clear();
        _shuffled_species.clear();
        _shuffle_keys.clear();
        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    const HostBuffer<Unit<T>> units(_units.begin(), _units.end());
    const HostBuffer<SpawnOperator<T>> spawn_operators(
        _spawn_operators.begin(),
        _spawn_operators.end());

    _local_positions.clear();
    _local_positions.resize(units.size());

    std::size_t total_count                = 0;
    const std::size_t spawn_operator_count = spawn_operators.size();

    for (std::size_t i = 0; i < units.size(); ++i) {
        const auto& geometry = units[i].geometry_operator();

        const auto bounds = geometry.bound();

        const auto& lower = bounds.lower_corner;
        const auto& upper = bounds.upper_corner;

        const auto& spawn_op = spawn_operators[(spawn_operator_count == 1) ? 0 : i];

        const int nx = atlas::sampling::sample_axis_count(lower.x, upper.x, _spacing);
        const int ny = atlas::sampling::sample_axis_count(lower.y, upper.y, _spacing);
        const int nz = atlas::sampling::sample_axis_count(lower.z, upper.z, _spacing);

        HostBuffer<Vector3<T>> positions;

        if (nx > 0 && ny > 0 && nz > 0) {
            positions.reserve(
                static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));

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

        _local_positions[i] = DeviceBuffer<Vector3<T>>(positions.begin(), positions.end());
    }

    _species_cache.clear();
    _shuffled_species.clear();
    _shuffle_keys.clear();

    if (total_count == 0) {

        _shuffle_seed         = 0;
        _is_invalidated_cache = false;
        return;
    }

    _species_cache.resize(total_count);
    _shuffled_species.resize(total_count);
    _shuffle_keys.resize(total_count);

    const int count         = static_cast<int>(total_count);
    const int species_count = static_cast<int>(mole_fractions.size());
    const auto* fractions   = atlas::raw_pointer_cast(mole_fractions.data());
    auto* cache             = atlas::raw_pointer_cast(this->_species_cache.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_DEVICE(const int i) {
            const T fraction = static_cast<T>(i) / static_cast<T>(count);

            T sum          = T(0);
            size_t species = 0;

            for (int s = 0; s < species_count; ++s) {
                sum += fractions[s];
                if (fraction < sum) {
                    species = static_cast<size_t>(s);
                    break;
                }
            }

            cache[i] = species;
        });

    _shuffle_seed         = 0;
    _is_invalidated_cache = false;
}

template <typename T>
void
Source<T>::shuffle_species(const std::size_t count) {

    if (count == 0) {
        _shuffled_species.clear();
        _shuffle_keys.clear();
        return;
    }

    _shuffled_species = _species_cache;

    const std::uint64_t seed = _shuffle_seed++;

    auto* keys = atlas::raw_pointer_cast(this->_shuffle_keys.data());
    const ShuffleOperator shuffle {};

    atlas::parallel_for<ExecutionPolicy::device>(
        static_cast<std::size_t>(0),
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            keys[i] = shuffle(static_cast<int>(i), seed);
        });

    atlas::parallel_sort_by_key<ExecutionPolicy::device>(
        _shuffle_keys.begin(),
        _shuffle_keys.end(),
        _shuffled_species.begin());
}

template <typename T>
void
Source<T>::emit(FluidDeviceProbe<T>& particle_probe) {

    rebuild_cache();

    if (_units.empty() || !_fluid || _fluid->empty() || _local_positions.empty()) {
        return;
    }

    const auto& generators_buf = _fluid->generators();
    const auto& particles_buf  = _fluid->particles();

    if (generators_buf.empty() || particles_buf.empty()) {
        return;
    }

    std::size_t total_count = 0;
    for (const auto& positions : _local_positions) {
        total_count += positions.size();
    }

    if (total_count == 0) {
        return;
    }

    const int begin = particle_probe.particle_count;
    const int end   = begin + static_cast<int>(total_count);

    if (end > static_cast<int>(particle_probe.buffer_size)) {
        atlas::logger::warn()
            << "Source::emit: insufficient space. Required: " << end
            << ", Available: " << particle_probe.buffer_size;
        return;
    }

    shuffle_species(total_count);

    const auto* units      = atlas::raw_pointer_cast(this->_units.data());
    const auto* generators = atlas::raw_pointer_cast(generators_buf.data());
    const auto* particles  = atlas::raw_pointer_cast(particles_buf.data());
    const auto* species    = atlas::raw_pointer_cast(this->_shuffled_species.data());
    const T temperature    = _temperature;

    std::size_t species_offset = 0;
    int dst_offset             = begin;

    for (std::size_t unit_index = 0; unit_index < _local_positions.size(); ++unit_index) {

        const auto& positions = _local_positions[unit_index];
        const int count       = static_cast<int>(positions.size());

        if (count == 0) {
            continue;
        }

        const auto* local_positions = atlas::raw_pointer_cast(positions.data());
        const auto* local_species   = species + species_offset;

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            count,
            [=] ATLAS_DEVICE(const int i) {
                const int dst = dst_offset + i;

                const size_t sid = local_species[i];

                Vector3<T> world_pos;
                units[unit_index].sync_operator().sync_to_world(local_positions[i], world_pos);

                particle_probe.pos[dst] = world_pos;

                particle_probe.vel[dst] = generators[sid].generate(temperature, particles[sid].mass);

                particle_probe.temperature[dst] = temperature;

                particle_probe.species[dst] = sid;
            });

        species_offset += positions.size();

        dst_offset += count;
    }

    particle_probe.particle_count = end;

    atlas::logger::info() << "\n"
                          << "Source::emit: emitted " << total_count << " particles.";
}
template <typename T>
void
Source<T>::set_units(DeviceBuffer<Unit<T>> units) noexcept {

    _units                = std::move(units);
    _is_invalidated_cache = true;
}

template <typename T>
void
Source<T>::set_units(const HostBuffer<Unit<T>>& units) {

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
Source<T>::set_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

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

    _spawn_types          = DeviceBuffer<SpawnType>(spawn_types.begin(), spawn_types.end());
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

    _spawn_operators      = DeviceBuffer<SpawnOperator<T>>(spawn_operators.begin(), spawn_operators.end());
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
const atlas::host_shared_ptr<atlas::Fluid<T>>&
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
const HostBuffer<DeviceBuffer<Vector3<T>>>&
Source<T>::local_positions() const noexcept {

    return _local_positions;
}

template <typename T>
Source<T>
Source<T>::Builder::build() {

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
Source<T>::Builder::with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept {

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

    if (!std::isfinite(_spacing) || _spacing <= T(0)) {

        atlas::logger::error()
            << "Source::Builder: spacing must be finite and positive.";
        throw std::runtime_error("Source::Builder: spacing must be finite and positive.");
    }

    if (!std::isfinite(_tolerance)) {

        atlas::logger::error()
            << "Source::Builder: tolerance must be finite.";
        throw std::runtime_error("Source::Builder: tolerance must be finite.");
    }

    if (!std::isfinite(_temperature)) {

        atlas::logger::error()
            << "Source::Builder: temperature must be finite.";
        throw std::runtime_error("Source::Builder: temperature must be finite.");
    }
}

}