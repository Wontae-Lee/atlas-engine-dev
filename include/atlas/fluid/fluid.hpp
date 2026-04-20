#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_fill.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/scan/exclusive_scan.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace atlas::fluid {

template <typename T>
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {

    // Return a default-initialized builder that can be configured
    // through the fluent Builder API before constructing a Fluid object.
    return Builder {};
}

template <typename T>
Fluid<T>::Fluid(const size_t buffer_size)
    : _buffer_size(buffer_size) {
    // Pre-register the default states required by the fluid system.
    //
    // These states are allocated with the full buffer capacity so they can
    // store per-particle data for any slot in the fluid buffer:
    // - position: particle positions
    // - velocity: particle velocities
    // - species : material/species identifiers
    // - active  : liveness flag used for reuse/compaction
    //
    // The constructor only prepares storage; it does not create active
    // particles yet.
    emplace_state<FluidPositionState<T>>(buffer_size);
    emplace_state<FluidVelocityState<T>>(buffer_size);
    emplace_state<FluidSpeciesState<T>>(buffer_size);
    emplace_state<FluidActiveState<T>>(buffer_size);
    _keep.resize(buffer_size);
    _offsets.resize(buffer_size);
    _compact_indices.resize(buffer_size);
}

template <typename T>
const DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() const noexcept {

    // Expose the generator operator buffer as a read-only reference.
    // Each entry typically corresponds to the logic used to emit or initialize
    // particles for a specific particle/material source.
    return _generators;
}

template <typename T>
DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() noexcept {

    // Expose the generator operator buffer as a mutable reference so callers
    // can update or replace generation logic after construction.
    return _generators;
}

template <typename T>
const DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particle_properties() const noexcept {

    // Expose the particle-property buffer so kinetic solvers can access
    // per-species physical parameters such as mass and collision diameter.
    return _particle_properties;
}

template <typename T>
DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particle_properties() noexcept {

    // Mutable overload of particle_properties() for callers that need to
    // update the installed material table after construction.
    return _particle_properties;
}

template <typename T>
size_t
Fluid<T>::particle_count() const noexcept {
    // Return the current number of logically active particles.
    //
    // This is not necessarily equal to the full buffer capacity because the
    // allocation is fixed-size and may contain inactive or reusable slots.
    return _particle_count;
}

template <typename T>
T
Fluid<T>::statistical_weight() const noexcept {
    // Return the statistical weight of the fluid.
    return _statistical_weight;
}

template <typename T>
void
Fluid<T>::remove_particles() {
    // The active state determines whether each particle slot is currently used.
    // If the active state does not exist, there is no reliable way to determine
    // which particles must be removed or kept, so the operation becomes a no-op.
    auto* active_state = state<FluidActiveState<T>>();
    if (active_state == nullptr) {
        return;
    }

    auto& active     = active_state->data();
    const auto count = _buffer_size;

    if (count == 0) {
        _particle_count = 0;
        return;
    }

    if (_keep.size() != count) {
        _keep.resize(count);
    }

    if (_offsets.size() != count) {
        _offsets.resize(count);
    }

    if (_compact_indices.size() != count) {
        _compact_indices.resize(count);
    }

    // Build a 0/1 keep mask from the active state:
    //   active == 0 -> remove this particle
    //   active != 0 -> keep this particle
    //
    // The keep mask is written into a separate buffer because the active buffer
    // itself is one of the states that will later participate in compaction.
    // Keeping the mask separate avoids overwriting information too early.
    const auto* active_ptr = atlas::raw_pointer_cast(active.data());
    auto* keep_ptr         = atlas::raw_pointer_cast(this->_keep.data());

    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            // Normalize the active flag into a strict binary keep mask.
            //
            // Any non-zero value is treated as "keep", which makes the code
            // robust even if the active state stores values other than 0/1.
            keep_ptr[i] = active_ptr[i] == 0 ? std::size_t { 0 } : std::size_t { 1 };
        });

    // Compute an exclusive prefix sum over the keep mask.
    //
    // For each source index i:
    // - offsets[i] tells where that particle should be written in the compacted
    //   destination range if keep[i] == 1
    // - keep[i] tells whether that source particle survives at all
    //
    // Example:
    //   keep    = [1, 0, 1, 1]
    //   offsets = [0, 1, 1, 2]
    //
    // Interpretation:
    //   source 0 survives and moves to destination 0
    //   source 1 is removed
    //   source 2 survives and moves to destination 1
    //   source 3 survives and moves to destination 2
    atlas::exclusive_scan<ExecutionPolicy::device>(
        _keep.begin(),
        _keep.begin() + static_cast<std::ptrdiff_t>(count),
        _offsets.begin(),
        std::size_t { 0 });

    // exclusive_scan produces destination offsets but does not directly report
    // how many particles survive in total.
    //
    // The final number of kept particles is:
    //   last_offset + last_keep
    //
    // because:
    // - last_offset is the number of kept items before the last element
    // - last_keep adds 1 if the last element survives, otherwise 0
    const std::size_t kept = _offsets[count - 1] + _keep[count - 1];

    // Fast path: if every particle is kept, the logical particle count remains
    // equal to the full count and no compaction work is needed.
    if (kept == count) {
        _particle_count = kept;
        return;
    }

    // Fast path: if no particles survive, there is no need to build gather
    // indices or compact every state.
    //
    // The backing storage remains allocated at full capacity. We only clear the
    // active state so all slots become reusable for later particle emission.
    if (kept == 0) {
        atlas::parallel_fill<ExecutionPolicy::device>(
            active.begin(),
            active.begin() + static_cast<std::ptrdiff_t>(count),
            0);
        _particle_count = 0;
        return;
    }

    const auto* offsets_ptr = atlas::raw_pointer_cast(this->_offsets.data());
    auto* indices_ptr       = atlas::raw_pointer_cast(this->_compact_indices.data());

    // Convert the keep/offset representation into a direct compacted index map:
    //
    //   compact_indices[destination] = source
    //
    // Example:
    //   keep            = [1, 0, 1, 1]
    //   offsets         = [0, 1, 1, 2]
    //   compact_indices = [0, 2, 3]
    //
    // This means:
    //   destination 0 reads from source 0
    //   destination 1 reads from source 2
    //   destination 2 reads from source 3
    //
    // A single shared index map allows every state buffer to be compacted in
    // exactly the same way, guaranteeing that all per-particle attributes remain
    // aligned after removal.
    atlas::parallel_for<ExecutionPolicy::device>(
        std::size_t { 0 },
        count,
        [=] ATLAS_DEVICE(const std::size_t i) {
            if (keep_ptr[i] != 0) {
                indices_ptr[offsets_ptr[i]] = i;
            }
        });

    // Compact every registered state using the same source-index mapping.
    //
    // This is the key design point:
    // - position, velocity, species, active, and any custom future state all
    //   move consistently
    // - particle attributes remain synchronized after dead particles are removed
    // - compaction logic for each state is centralized behind FluidState::compact
    for (auto& state : _states | std::views::values) {
        if (state) {
            state->compact(_compact_indices, kept);
        }
    }

    // After compaction, all valid particles are packed into the prefix [0, kept).
    //
    // The remaining tail [kept, count) still exists physically because the
    // buffer is fixed-capacity. Those slots are now logically unused and must
    // be marked inactive so future emitters can safely recycle them.
    //
    // Only the active flags in the tail are cleared here. Other state buffers in
    // that tail do not need to be erased because they are outside the active
    // particle range and therefore ignored by the simulation.
    atlas::parallel_fill<ExecutionPolicy::device>(
        active.begin() + static_cast<std::ptrdiff_t>(kept),
        active.begin() + static_cast<std::ptrdiff_t>(count),
        0);

    // Update the logical number of active particles after compaction.
    _particle_count = kept;
}

template <typename T>
void
Fluid<T>::set_particle_count(const size_t particle_count) {
    // The active prefix must remain within the allocated storage.
    if (particle_count > _buffer_size) {
        throw std::out_of_range("Fluid::set_particle_count: particle_count exceeds buffer_size.");
    }

    _particle_count = particle_count;
}

template <typename T>
template <typename StateT, typename... Args>
StateT&
Fluid<T>::emplace_state(Args&&... args) {
    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    // Construct the requested state in-place using perfect forwarding so the
    // caller can pass arbitrary constructor arguments efficiently.
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);

    // Keep a raw pointer before transferring ownership so we can return a stable
    // reference to the stored object after insertion.
    auto* ptr = state.get();

    // Insert the state by its exact dynamic type. If a state of the same type
    // already exists, it is replaced.
    _states.insert_or_assign(typeid(StateT), std::move(state));

    // Return a reference to the newly stored state for immediate use.
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Fluid<T>::set_state(std::unique_ptr<StateT> state) {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    // Reject null ownership transfer because the API contract expects a valid
    // state object to be installed.
    if (state == nullptr) {
        throw std::invalid_argument("Fluid::set_state failed: state must not be null.");
    }

    // Store or replace the state keyed by its type.
    _states.insert_or_assign(typeid(StateT), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Fluid<T>::state() noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    // Look up the requested state by its concrete type.
    auto it = _states.find(typeid(StateT));

    // Return nullptr if the state is not registered; otherwise cast the base
    // pointer back to the requested derived type.
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Fluid<T>::state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    // Const-qualified overload of state(), preserving read-only access to the
    // returned state object.
    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Fluid<T>::has_state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    // Check whether a state of the requested type is currently registered.
    return _states.contains(typeid(StateT));
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Fluid<T>::remove_state() {
    static_assert(std::is_base_of_v<FluidState, StateT>, "StateT must derive from atlas::fluid::FluidState.");

    // Find the state by type. If it does not exist, return nullptr to indicate
    // that no state was removed.
    auto it = _states.find(typeid(StateT));
    if (it == _states.end()) {
        return nullptr;
    }

    // Release ownership from the internal base-class pointer, cast it back to
    // the requested derived type, and transfer ownership to the caller.
    auto state = std::unique_ptr<StateT>(static_cast<StateT*>(it->second.release()));

    // Remove the now-empty entry from the registry.
    _states.erase(it);

    return state;
}

template <typename T>
size_t
Fluid<T>::buffer_size() const noexcept {
    // Return the fixed storage capacity of this fluid object.
    //
    // This value is the maximum number of particle slots the fluid can manage
    // without reallocating or rebuilding its internal state buffers.
    return _buffer_size;
}

template <typename T>
Fluid<T>
Fluid<T>::Builder::build() const {

    // Validate the builder configuration before constructing the Fluid.
    //
    // This ensures that related arrays that must match structurally already
    // satisfy the expected invariants.
    validate();

    Fluid<T> f {};

    // Copy particle material/property data from the builder-owned device buffer
    // into the Fluid instance.
    f._particle_properties = DeviceBuffer<MatrialProperties<T>>(_particles.begin(), _particles.end());

    // Copy generator operators into the Fluid instance.
    f._generators = DeviceBuffer<GenerateOperator<T>>(_generators.begin(), _generators.end());

    // Set the fixed particle capacity of the created Fluid.
    f._buffer_size        = _buffer_size;
    f._statistical_weight = _statistical_weight;
    f._keep.resize(_buffer_size);
    f._offsets.resize(_buffer_size);
    f._compact_indices.resize(_buffer_size);

    // Install the default runtime states required by the simulation.
    //
    // Each state is sized to the full buffer capacity so it can hold one entry
    // per possible particle slot.
    f.emplace_state<FluidPositionState<T>>(_buffer_size);
    f.emplace_state<FluidVelocityState<T>>(_buffer_size);
    f.emplace_state<FluidSpeciesState<T>>(_buffer_size);
    f.emplace_state<FluidActiveState<T>>(_buffer_size);

    return f;
}

template <typename T>
atlas::host_shared_ptr<Fluid<T>>
Fluid<T>::Builder::make_host_shared() const {
    // Build a temporary Fluid value first, then move it into a host-managed
    // shared allocation.
    //
    // This is a convenience API for users who want shared ownership directly
    // from the builder without writing the two-step build + make_host_shared
    // sequence themselves.
    auto f = build();
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_properties(const HostBuffer<MatrialProperties<T>>& properties) {
    // Upload particle/material property data from host memory into the builder's
    // device buffer so the final Fluid can be constructed entirely with device-
    // resident property storage.
    _particles = DeviceBuffer<MatrialProperties<T>>(properties.begin(), properties.end());
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators) {
    // Rebuild the generator operator buffer from the provided host-side
    // generator objects.
    //
    // Each host generator is converted into its device-executable operator form.
    // Null generator entries are replaced with a default-constructed operator so
    // the resulting buffer length stays aligned with the particle/property data.
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
    // Set the maximum number of particles the final Fluid instance can hold.
    //
    // This affects the size of all default per-particle state buffers allocated
    // during construction.
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
void
Fluid<T>::Builder::validate() const {
    // The builder currently requires a one-to-one correspondence between
    // particle property entries and generator operator entries.
    //
    // A mismatch would mean the builder cannot safely determine which generator
    // belongs to which particle/material descriptor.
    if (_particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/generators size mismatch.");
    }

    if (!(_statistical_weight > T(0))) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }

    for (std::size_t i = 0; i < _particles.size(); ++i) {
        const MatrialProperties<T> particle_property = _particles[i];

        if (!(particle_property.molecular_mass > T(0))) {
            throw std::runtime_error(
                "Fluid::Builder: molecular_mass must be positive.");
        }

        if (particle_property.mass != particle_property.molecular_mass * _statistical_weight) {
            throw std::runtime_error(
                "Fluid::Builder: particle mass does not match statistical weight.");
        }
    }
}

}
