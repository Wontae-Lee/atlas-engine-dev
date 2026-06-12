#pragma once
#include <atlas/serialization/protobuf_snapshot.h>
#include <stdexcept>
#include <type_traits>
#include <utility>
namespace atlas {
template <typename T>
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
Fluid<T>::Fluid(const size_t buffer_size)
    : _buffer_size(buffer_size) {
    _states.reserve(4);
    emplace_state<FluidPositionState<T>>(buffer_size);
    emplace_state<FluidVelocityState<T>>(buffer_size);
    emplace_state<FluidSpeciesState<T>>(buffer_size);
    emplace_state<FluidActiveState<T>>(buffer_size);
}

template <typename T>
const DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() const noexcept {
    return _generators;
}

template <typename T>
DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() noexcept {
    return _generators;
}

template <typename T>
const DeviceBuffer<MaterialProperties<T>>&
Fluid<T>::particle_properties() const noexcept {
    return _particle_properties;
}

template <typename T>
DeviceBuffer<MaterialProperties<T>>&
Fluid<T>::particle_properties() noexcept {
    return _particle_properties;
}

template <typename T>
size_t
Fluid<T>::particle_count() const noexcept {
    return _particle_count;
}

template <typename T>
T
Fluid<T>::statistical_weight() const noexcept {
    return _statistical_weight;
}

template <typename T>
void
Fluid<T>::save(const std::string_view path) const {
    atlas::save_fluid_binary(*this, path);
}

template <typename T>
const ObserverHostPtr&
Fluid<T>::observer() const noexcept {
    return _observer;
}

template <typename T>
template <typename StateT>
const std::type_index&
Fluid<T>::state_key() noexcept {
    static const std::type_index key { typeid(StateT) };
    return key;
}

template <typename T>
void
Fluid<T>::set_particle_count(const size_t particle_count) {
    if (particle_count > _buffer_size) {
        throw std::out_of_range("Fluid::set_particle_count: particle_count exceeds buffer_size.");
    }
    _particle_count = particle_count;
}

template <typename T>
template <typename StateT, typename... Args>
StateT&
Fluid<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::FluidState.");
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);
    auto* ptr  = state.get();
    _states.insert_or_assign(state_key<StateT>(), std::move(state));
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Fluid<T>::set_state(std::unique_ptr<StateT> state) {
    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::FluidState.");
    if (state == nullptr) {
        throw std::invalid_argument("Fluid::set_state failed: state must not be null.");
    }
    _states.insert_or_assign(state_key<StateT>(), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Fluid<T>::state() noexcept {
    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::FluidState.");
    const auto& key = state_key<StateT>();
    auto it = _states.find(key);
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Fluid<T>::state() const noexcept {
    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::FluidState.");
    const auto& key = state_key<StateT>();
    auto it = _states.find(key);
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Fluid<T>::has_state() const noexcept {
    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::FluidState.");
    return _states.contains(state_key<StateT>());
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Fluid<T>::remove_state() {
    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::FluidState.");
    const auto& key = state_key<StateT>();
    auto it = _states.find(key);
    if (it == _states.end()) {
        return nullptr;
    }
    auto state = std::unique_ptr<StateT>(static_cast<StateT*>(it->second.release()));
    _states.erase(it);
    return state;
}

template <typename T>
std::unordered_map<std::type_index, std::unique_ptr<FluidState>>&
Fluid<T>::states() noexcept {
    return _states;
}

template <typename T>
const std::unordered_map<std::type_index, std::unique_ptr<FluidState>>&
Fluid<T>::states() const noexcept {
    return _states;
}

template <typename T>
size_t
Fluid<T>::buffer_size() const noexcept {
    return _buffer_size;
}

template <typename T>
Fluid<T>
Fluid<T>::Builder::build() const {
    validate();
    Fluid<T> f(_buffer_size);
    if (_temperature_state.has_value()) {
        f._states.reserve(5);
    }
    f._particle_properties = DeviceBuffer<MaterialProperties<T>>(_particles.begin(), _particles.end());
    f._generators          = DeviceBuffer<GenerateOperator<T>>(_generators.begin(), _generators.end());
    f._statistical_weight  = _statistical_weight;
    f._observer            = _observer;
    if (_position_state.has_value()) {
        f.template set_state<FluidPositionState<T>>(std::make_unique<FluidPositionState<T>>(
            DeviceBuffer<Vector3<T>>(_position_state->begin(), _position_state->end())));
    }
    if (_velocity_state.has_value()) {
        f.template set_state<FluidVelocityState<T>>(std::make_unique<FluidVelocityState<T>>(
            DeviceBuffer<Vector3<T>>(_velocity_state->begin(), _velocity_state->end())));
    }
    if (_species_state.has_value()) {
        f.template set_state<FluidSpeciesState<T>>(std::make_unique<FluidSpeciesState<T>>(
            DeviceBuffer<std::size_t>(_species_state->begin(), _species_state->end())));
    }
    if (_active_state.has_value()) {
        f.template set_state<FluidActiveState<T>>(std::make_unique<FluidActiveState<T>>(
            DeviceBuffer<int>(_active_state->begin(), _active_state->end())));
    }
    if (_temperature_state.has_value()) {
        f.template set_state<FluidTemperatureState<T>>(std::make_unique<FluidTemperatureState<T>>(
            DeviceBuffer<T>(_temperature_state->begin(), _temperature_state->end())));
    }
    if (_particle_count.has_value()) {
        f.set_particle_count(*_particle_count);
    }
    return f;
}

template <typename T>
atlas::host_shared_ptr<Fluid<T>>
Fluid<T>::Builder::make_host_shared() const {
    auto f = build();
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_properties(const HostBuffer<MaterialProperties<T>>& properties) {
    _particles = properties;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators) {
    _generators.clear();
    const int n = static_cast<int>(generators.size());
    _generators.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        _generators.push_back(
            generators[i] ? generators[i]->make_generate_operator() : GenerateOperator<T> {});
    }
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_buffer_size(const size_t buffer_size) noexcept {
    _buffer_size = buffer_size;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_statistical_weight(const T statistical_weight) noexcept {
    _statistical_weight = statistical_weight;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_observer(ObserverHostPtr observer) noexcept {
    _observer = std::move(observer);
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_binary(const std::string& path) {
    auto snapshot       = atlas::load_fluid_binary<T>(path);
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

template <typename T>
void
Fluid<T>::Builder::validate() const {
    if (_particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/generators size mismatch.");
    }
    if (!(_statistical_weight > T(0))) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }
    for (std::size_t i = 0; i < _particles.size(); ++i) {
        const MaterialProperties<T>& particle_property = _particles[i];
        if (!(particle_property.molecular_mass > T(0))) {
            throw std::runtime_error(
                "Fluid::Builder: molecular_mass must be positive.");
        }
        const T expected_mass = particle_property.molecular_mass * _statistical_weight;
        if (particle_property.mass != expected_mass) {
            throw std::runtime_error(
                "Fluid::Builder: particle mass does not match statistical weight.");
        }
    }
}

}
