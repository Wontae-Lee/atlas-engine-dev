#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace atlas::system {

template <typename T>
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
Fluid<T>::Fluid(const size_t buffer_size)
    : _buffer_size(buffer_size) {
    emplace_state<ParticleState<T>>(buffer_size);
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
template <typename StateT, typename... Args>
StateT&
Fluid<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::system::FluidState.");
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);
    auto* ptr  = state.get();
    _states.insert_or_assign(typeid(StateT), std::move(state));
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Fluid<T>::set_state(std::unique_ptr<StateT> state) {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::system::FluidState.");
    if (state == nullptr) {
        throw std::invalid_argument("Fluid::set_state failed: state must not be null.");
    }

    _states.insert_or_assign(typeid(StateT), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Fluid<T>::state() noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::system::FluidState.");

    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Fluid<T>::state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::system::FluidState.");

    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Fluid<T>::has_state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::system::FluidState.");
    return _states.contains(typeid(StateT));
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Fluid<T>::remove_state() {
    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::system::FluidState.");
    auto it = _states.find(typeid(StateT));
    if (it == _states.end()) {
        return nullptr;
    }
    auto state = std::unique_ptr<StateT>(static_cast<StateT*>(it->second.release()));
    _states.erase(it);
    return state;
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
    Fluid<T> f {};
    f._particle_properties = DeviceBuffer<MatrialProperties<T>>(_particles.begin(), _particles.end());
    f._generators          = DeviceBuffer<GenerateOperator<T>>(_generators.begin(), _generators.end());
    f._buffer_size         = _buffer_size;
    f.emplace_state<ParticleState<T>>(_buffer_size);
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
Fluid<T>::Builder::with_properties(const HostBuffer<MatrialProperties<T>>& properties) {
    _particles = DeviceBuffer<MatrialProperties<T>>(properties.begin(), properties.end());
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators) {
    _generators.clear();
    const int n = static_cast<int>(generators.size());
    for (int i = 0; i < n; ++i) {
        _generators.push_back(generators[i] ? generators[i]->make_generate_operator() : GenerateOperator<T> {});
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
void
Fluid<T>::Builder::validate() const {
    if (_particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/generators size mismatch.");
    }
}

}