#include <atlas/fluid/fluid.h>

#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <stdexcept>
#include <utility>

namespace atlas {

namespace {

/**
 * @brief Copy an explicitly supplied live prefix into a fluid column.
 * @tparam State Concrete particle-state column.
 * @tparam Value Element stored by the column.
 * @param fluid Fluid receiving the staged prefix.
 * @param values Optional host-side values; absence leaves the column at its default.
 */
template <typename State, typename Value>
void initialize_state(Fluid& fluid, const std::optional<HostBuffer<Value>>& values) {
    if (!values) return;
    State* state = fluid.state<State>();
    if (state == nullptr) state = &fluid.emplace_state<State>(fluid.buffer_size());
    if (!values->empty()) atlas::copy_host_to_device(&values->front(), state->data(), values->size());
}

/**
 * @brief Reject a supplied column whose live prefix has the wrong length.
 * @tparam Value Element stored by the column.
 * @param values Optional staged values.
 * @param particle_count Required live prefix length.
 */
template <typename Value>
void validate_state(const std::optional<HostBuffer<Value>>& values, std::size_t particle_count) {
    if (values && values->size() != particle_count) {
        throw std::runtime_error("Fluid::Builder: initial state length must equal particle_count.");
    }
}

}

Fluid::Fluid(const std::size_t buffer_size)
    : _buffer_size(buffer_size)
    , _active(buffer_size) {
    // Seed exactly the three mandatory columns; optional ones are added on demand.
    _states.reserve(3);
    emplace_state<FluidPositionState>(buffer_size);
    emplace_state<FluidVelocityState>(buffer_size);
    emplace_state<FluidSpeciesState>(buffer_size);
}

Fluid::Fluid(const std::size_t buffer_size, MaterialDictionaryHostPtr materials)
    : Fluid(buffer_size) {
    _materials = std::move(materials);
}

Fluid::Builder
Fluid::builder() noexcept {
    return Builder {};
}

void
Fluid::set_particle_count(const std::size_t particle_count) {
    if (particle_count > _buffer_size) {
        throw std::out_of_range("Fluid::set_particle_count: particle_count exceeds buffer_size.");
    }
    _particle_count = particle_count;
}

DeviceBuffer<int>&
Fluid::active() noexcept {
    return _active;
}

const DeviceBuffer<int>&
Fluid::active() const noexcept {
    return _active;
}

std::size_t
Fluid::compact() {
    const auto alive = _particle_count;

    // Nothing to pack, or the flag buffer is too short to scan safely.
    if (alive == 0 || _active.size() < alive) {
        return alive;
    }

    const int particle_count = static_cast<int>(alive);

    if (_survivor_offsets.size() != alive) {
        _survivor_offsets.resize(alive);
    }

    // A survivor's prefix sum is its slot in the compacted buffers.
    const auto* flags = atlas::raw_pointer_cast(_active.data());

    atlas::exclusive_scan<ExecutionPolicy::device>(
        flags,
        flags + particle_count,
        _survivor_offsets.begin(),
        0);

    if (_survivor_total.size() < 1) {
        _survivor_total.resize(1);
    }

    auto* total         = atlas::raw_pointer_cast(_survivor_total.data());
    const auto* offsets = atlas::raw_pointer_cast(_survivor_offsets.data());
    const int last      = particle_count - 1;

    // Exclusive scan omits the last flag, so the total is the last offset plus that flag.
    // A single-thread kernel writes it to device memory to avoid a second scan pass.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        1,
        [=] ATLAS_ALL_DEVICE(int) {
            total[0] = offsets[last] + flags[last];
        });

    // The one host/device sync in this routine: the count drives the host-side control flow.
    int survivors = 0;
    atlas::copy_device_to_host(total, &survivors, 1);

    const auto kept = static_cast<std::size_t>(survivors < 0 ? 0 : survivors);

    // All alive: the buffers are already packed, so skip the gather entirely.
    if (kept == alive) {
        return kept;
    }

    if (kept > 0) {
        if (_compact_indices.size() != kept) {
            _compact_indices.resize(kept);
        }

        auto* compact_indices = atlas::raw_pointer_cast(_compact_indices.data());

        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            particle_count,
            [=] ATLAS_ALL_DEVICE(const int i) {
                if (flags[i] != 0) {
                    compact_indices[offsets[i]] = static_cast<std::size_t>(i);
                }
            });
    }

    // Every state holds one entry per particle, so they all gather by the same
    // indices.
    for (auto& [type, fluid_state] : _states) {
        fluid_state->compact(_compact_indices, kept);
    }

    // Whatever survived is alive by definition, so the flags need gathering no
    // more than a fill does.
    atlas::parallel_fill<ExecutionPolicy::device>(
        _active.begin(),
        _active.begin() + static_cast<std::ptrdiff_t>(kept),
        1);

    set_particle_count(kept);

    return kept;
}

FluidStateStore&
Fluid::states() noexcept {
    return _states;
}

const FluidStateStore&
Fluid::states() const noexcept {
    return _states;
}

std::size_t
Fluid::buffer_size() const noexcept {
    return _buffer_size;
}

std::size_t
Fluid::particle_count() const noexcept {
    return _particle_count;
}

float
Fluid::statistical_weight() const noexcept {
    return _statistical_weight;
}

const MaterialDictionaryHostPtr&
Fluid::materials() const noexcept {
    return _materials;
}

Fluid::Builder&
Fluid::Builder::with_buffer_size(const std::size_t buffer_size) noexcept {
    _buffer_size = buffer_size;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_particle_count(const std::size_t particle_count) noexcept {
    _particle_count = particle_count;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_statistical_weight(const float statistical_weight) noexcept {
    _statistical_weight = statistical_weight;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_materials(MaterialDictionaryHostPtr materials) noexcept {
    _materials = std::move(materials);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_position(HostBuffer<Float3> values) {
    _position = std::move(values);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_velocity(HostBuffer<Float3> values) {
    _velocity = std::move(values);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_species(HostBuffer<std::size_t> values) {
    _species = std::move(values);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_temperature(HostBuffer<float> values) {
    _temperature = std::move(values);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_translational_energy(HostBuffer<float> values) {
    _translational_energy = std::move(values);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_rotational_energy(HostBuffer<float> values) {
    _rotational_energy = std::move(values);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_vibrational_energy(HostBuffer<float> values) {
    _vibrational_energy = std::move(values);
    return *this;
}

void
Fluid::Builder::validate() const {
    if (!(_statistical_weight > 0.0f)) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }
    if (_particle_count > _buffer_size) {
        throw std::runtime_error(
            "Fluid::Builder: particle_count exceeds buffer_size.");
    }
    validate_state(_position, _particle_count);
    validate_state(_velocity, _particle_count);
    validate_state(_species, _particle_count);
    validate_state(_temperature, _particle_count);
    validate_state(_translational_energy, _particle_count);
    validate_state(_rotational_energy, _particle_count);
    validate_state(_vibrational_energy, _particle_count);
    if (_materials && _species) {
        for (const std::size_t species : *_species) {
            if (species >= _materials->materials().size()) {
                throw std::runtime_error("Fluid::Builder: species id is outside the material dictionary.");
            }
        }
    }
}

Fluid
Fluid::Builder::build() const {
    validate();

    Fluid fluid(_buffer_size, _materials);
    fluid._statistical_weight = _statistical_weight;
    fluid.set_particle_count(_particle_count);

    initialize_state<FluidPositionState>(fluid, _position);
    initialize_state<FluidVelocityState>(fluid, _velocity);
    initialize_state<FluidSpeciesState>(fluid, _species);
    initialize_state<FluidTemperatureState>(fluid, _temperature);
    initialize_state<FluidTranslationalEnergyState>(fluid, _translational_energy);
    initialize_state<FluidRotationalEnergyState>(fluid, _rotational_energy);
    initialize_state<FluidVibrationalEnergyState>(fluid, _vibrational_energy);

    return fluid;
}

atlas::host_unique_ptr<Fluid>
Fluid::Builder::make_host_unique() const {
    return atlas::make_host_unique<Fluid>(build());
}

}
