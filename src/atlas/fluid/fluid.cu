#include <atlas/fluid/fluid.h>

#include <atlas/serialization/protobuf_snapshot.h>

#include <memory>
#include <stdexcept>
#include <utility>

namespace atlas {

Fluid::Fluid(const std::size_t buffer_size)
    : _buffer_size(buffer_size) {
    _states.reserve(4);
    emplace_state<FluidPositionState>(buffer_size);
    emplace_state<FluidVelocityState>(buffer_size);
    emplace_state<FluidSpeciesState>(buffer_size);
    emplace_state<FluidActiveState>(buffer_size);
}

Fluid::Builder
Fluid::builder() noexcept {
    return Builder {};
}

const DeviceBuffer<Generate>&
Fluid::generators() const noexcept {
    return _generators;
}

DeviceBuffer<Generate>&
Fluid::generators() noexcept {
    return _generators;
}

const DeviceBuffer<MaterialProperties>&
Fluid::particle_properties() const noexcept {
    return _particle_properties;
}

DeviceBuffer<MaterialProperties>&
Fluid::particle_properties() noexcept {
    return _particle_properties;
}

void
Fluid::set_particle_count(const std::size_t particle_count) {
    if (particle_count > _buffer_size) {
        throw std::out_of_range("Fluid::set_particle_count: particle_count exceeds buffer_size.");
    }
    _particle_count = particle_count;
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

const ObserverHostPtr&
Fluid::observer() const noexcept {
    return _observer;
}

void
Fluid::save(const std::string_view path) const {
    atlas::save_fluid_binary(*this, path);
}

Fluid
Fluid::Builder::build() const {
    validate();
    Fluid f(_buffer_size);
    if (_temperature_state.has_value()) {
        f._states.reserve(5);
    }
    f._particle_properties = DeviceBuffer<MaterialProperties>(_particles.begin(), _particles.end());
    f._generators          = DeviceBuffer<Generate>(_generators.begin(), _generators.end());
    f._statistical_weight  = _statistical_weight;
    f._observer            = _observer;
    if (_position_state.has_value()) {
        f.set_state<FluidPositionState>(std::make_unique<FluidPositionState>(
            DeviceBuffer<Float3>(_position_state->begin(), _position_state->end())));
    }
    if (_velocity_state.has_value()) {
        f.set_state<FluidVelocityState>(std::make_unique<FluidVelocityState>(
            DeviceBuffer<Float3>(_velocity_state->begin(), _velocity_state->end())));
    }
    if (_species_state.has_value()) {
        f.set_state<FluidSpeciesState>(std::make_unique<FluidSpeciesState>(
            DeviceBuffer<std::size_t>(_species_state->begin(), _species_state->end())));
    }
    if (_active_state.has_value()) {
        f.set_state<FluidActiveState>(std::make_unique<FluidActiveState>(
            DeviceBuffer<int>(_active_state->begin(), _active_state->end())));
    }
    if (_temperature_state.has_value()) {
        f.set_state<FluidTemperatureState>(std::make_unique<FluidTemperatureState>(
            DeviceBuffer<float>(_temperature_state->begin(), _temperature_state->end())));
    }
    if (_particle_count.has_value()) {
        f.set_particle_count(*_particle_count);
    }
    return f;
}

atlas::host_shared_ptr<Fluid>
Fluid::Builder::make_host_shared() const {
    auto f = build();
    return atlas::make_host_shared<Fluid>(std::move(f));
}

Fluid::Builder&
Fluid::Builder::with_properties(const HostBuffer<MaterialProperties>& properties) {
    _particles = properties;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_generators(const HostBuffer<GeneratorHostPtr>& generators) {
    _generators.clear();
    const int n = static_cast<int>(generators.size());
    _generators.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        _generators.push_back(
            generators[i] ? generators[i]->make_generate_operator() : Generate {});
    }
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_buffer_size(const std::size_t buffer_size) noexcept {
    _buffer_size = buffer_size;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_statistical_weight(const float statistical_weight) noexcept {
    _statistical_weight = statistical_weight;
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
    return *this;
}

Fluid::Builder&
Fluid::Builder::with_binary(const std::string& path) {
    auto snapshot       = atlas::load_fluid_binary(path);
    _buffer_size        = snapshot.buffer_size;
    _particle_count     = snapshot.particle_count;
    _statistical_weight = snapshot.statistical_weight;
    _particles          = std::move(snapshot.properties);
    _generators         = std::move(snapshot.generators);
    _position_state     = std::move(snapshot.positions);
    _velocity_state     = std::move(snapshot.velocities);
    _species_state      = std::move(snapshot.species);
    _active_state       = std::move(snapshot.active);
    _temperature_state  = std::move(snapshot.temperature);
    return *this;
}

void
Fluid::Builder::validate() const {
    if (_particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/generators size mismatch.");
    }
    if (!(_statistical_weight > 0.0f)) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }
    for (std::size_t i = 0; i < _particles.size(); ++i) {
        const MaterialProperties& particle_property = _particles[i];
        if (!(particle_property.molecular_mass > 0.0f)) {
            throw std::runtime_error(
                "Fluid::Builder: molecular_mass must be positive.");
        }
        const float expected_mass = particle_property.molecular_mass * _statistical_weight;
        if (particle_property.mass != expected_mass) {
            throw std::runtime_error(
                "Fluid::Builder: particle mass does not match statistical weight.");
        }
    }
}

}
