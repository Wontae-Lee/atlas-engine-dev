#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace atlas::fluid {

template <typename T>
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
Fluid<T>::Fluid(const size_t buffer_size)
    : _buffer_size(buffer_size) {
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
void
Fluid<T>::remove_particles() {
    auto* active_state = state<FluidActiveState<T>>();
    if (active_state == nullptr) {
        return;
    }

    auto& active     = active_state->data();
    const auto count = _buffer_size;

    // Build a 0/1 keep mask from the active state:
    //   active == 0 -> remove this particle
    //   active != 0 -> keep this particle
    // The keep mask is kept separate from active so the active buffer can be
    // compacted together with every other state later.
    const auto* active_ptr = atlas::raw_pointer_cast(active.data());
    auto* keep_ptr         = atlas::raw_pointer_cast(this->_keep.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            keep_ptr[i] = active_ptr[i] == 0 ? std::size_t { 0 } : std::size_t { 1 };
        });

    // Prefix-sum the keep mask. For each source index i, _offsets[i] becomes
    // the destination index where that particle should move if keep[i] == 1.
    // Example:
    //   keep    = [1, 0, 1, 1]
    //   offsets = [0, 1, 1, 2]
    //   source 0 -> dst 0, source 2 -> dst 1, source 3 -> dst 2
    atlas::exclusive_scan<ExecutionPolicy::device>(
        _keep.begin(),
        _keep.begin() + static_cast<std::ptrdiff_t>(count),
        _offsets.begin(),
        std::size_t { 0 });

    // exclusive_scan does not directly return the number of kept particles.
    // The final count is last_offset + last_keep.
    const std::size_t kept = _offsets[count - 1] + _keep[count - 1];
    if (kept == count) {
        return;
    }

    // If no particles survived, no state gather is needed. The buffers keep
    // their fixed capacity, so clearing the active prefix is enough to mark all
    // slots as reusable.
    if (kept == 0) {
        atlas::parallel_fill<ExecutionPolicy::device>(
            active.begin(),
            active.begin() + static_cast<std::ptrdiff_t>(count),
            0);
        return;
    }

    const auto* offsets_ptr = atlas::raw_pointer_cast(this->_offsets.data());
    auto* indices_ptr       = atlas::raw_pointer_cast(this->_compact_indices.data());

    // Convert the prefix-sum result into a compact source-index map:
    //   compact_indices[dst] = source
    // With the example above:
    //   compact_indices = [0, 2, 3]
    // Every state then uses this same mapping, which keeps position, velocity,
    // species, active, and any future state aligned without recomputing scan.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (keep_ptr[i] != 0) {
                indices_ptr[offsets_ptr[i]] = i;
            }
        });

    for (auto& [_, state] : _states) {
        if (state) {
            state->compact(_compact_indices, kept);
        }
    }

    // After gather, valid particles occupy [0, kept). The allocation still has
    // fixed capacity, so the tail [kept, count) is not erased or resized. Mark
    // only its active flags as inactive so emitters can reuse those slots.
    atlas::parallel_fill<ExecutionPolicy::device>(
        active.begin() + static_cast<std::ptrdiff_t>(kept),
        active.begin() + static_cast<std::ptrdiff_t>(count),
        0);
}

template <typename T>
template <typename StateT, typename... Args>
StateT&
Fluid<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);
    auto* ptr  = state.get();
    _states.insert_or_assign(typeid(StateT), std::move(state));
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Fluid<T>::set_state(std::unique_ptr<StateT> state) {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");
    if (state == nullptr) {
        throw std::invalid_argument("Fluid::set_state failed: state must not be null.");
    }

    _states.insert_or_assign(typeid(StateT), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Fluid<T>::state() noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Fluid<T>::state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Fluid<T>::has_state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");
    return _states.contains(typeid(StateT));
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Fluid<T>::remove_state() {
    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");
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
    f.emplace_state<FluidPositionState<T>>(_buffer_size);
    f.emplace_state<FluidVelocityState<T>>(_buffer_size);
    f.emplace_state<FluidSpeciesState<T>>(_buffer_size);
    f.emplace_state<FluidActiveState<T>>(_buffer_size);
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
