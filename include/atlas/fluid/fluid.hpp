#pragma once

#include <atlas/serialization/protobuf_snapshot.h>

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
    // - active  : liveness or occupancy flag used for reuse/compaction
    //
    // The constructor prepares storage only. It does not create active
    // particles yet.
    emplace_state<FluidPositionState<T>>(buffer_size);
    emplace_state<FluidVelocityState<T>>(buffer_size);
    emplace_state<FluidSpeciesState<T>>(buffer_size);
    emplace_state<FluidActiveState<T>>(buffer_size);
}

template <typename T>
const DeviceBuffer<GenerateOperator<T>>&
Fluid<T>::generators() const noexcept {

    // Expose the generator operator buffer as a read-only reference.
    //
    // Each entry typically contains device-executable logic used to generate,
    // emit, or initialize particles for a specific source or material entry.
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

    // Expose the particle/species property buffer so solvers can access
    // per-species physical parameters such as mass-related values.
    return _particle_properties;
}

template <typename T>
DeviceBuffer<MatrialProperties<T>>&
Fluid<T>::particle_properties() noexcept {

    // Mutable overload of particle_properties() for callers that need to
    // modify the installed property table after construction.
    return _particle_properties;
}

template <typename T>
size_t
Fluid<T>::particle_count() const noexcept {

    // Return the current number of logically active particles.
    //
    // This is not necessarily equal to the full buffer capacity because the
    // fluid uses fixed-capacity storage and may contain inactive or reusable slots.
    return _particle_count;
}

template <typename T>
T
Fluid<T>::statistical_weight() const noexcept {

    // Return the fluid-level statistical weight used by the simulation model.
    return _statistical_weight;
}

template <typename T>
void
Fluid<T>::save(const std::string_view path) const {
    atlas::serialization::save_fluid_binary(*this, path);
}

template <typename T>
const ObserverHostPtr&
Fluid<T>::observer() const noexcept {
    return _observer;
}

template <typename T>
void
Fluid<T>::set_particle_count(const size_t particle_count) {

    // Enforce that the active particle prefix never exceeds allocated capacity.
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
                  "StateT must derive from atlas::fluid::FluidState.");

    // Construct the requested state using perfect forwarding so the caller
    // can pass arbitrary constructor arguments efficiently.
    auto state = std::make_unique<StateT>(std::forward<Args>(args)...);

    // Keep a raw pointer before ownership transfer so a stable reference can
    // be returned after insertion into the registry.
    auto* ptr = state.get();

    // Insert the state keyed by its exact concrete type.
    // If the same type already exists, replace it.
    _states.insert_or_assign(typeid(StateT), std::move(state));

    // Return a reference to the newly stored state.
    return *ptr;
}

template <typename T>
template <typename StateT>
void
Fluid<T>::set_state(std::unique_ptr<StateT> state) {

    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::fluid::FluidState.");

    // Reject null ownership transfer because the API contract requires
    // a valid state object to be installed.
    if (state == nullptr) {
        throw std::invalid_argument("Fluid::set_state failed: state must not be null.");
    }

    // Store or replace the state keyed by its exact concrete type.
    _states.insert_or_assign(typeid(StateT), std::move(state));
}

template <typename T>
template <typename StateT>
StateT*
Fluid<T>::state() noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::fluid::FluidState.");

    // Look up the requested state by its exact concrete type.
    auto it = _states.find(typeid(StateT));

    // Return nullptr if the state is not registered; otherwise cast the base
    // pointer back to the requested derived type.
    return it == _states.end() ? nullptr : static_cast<StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
const StateT*
Fluid<T>::state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::fluid::FluidState.");

    // Const-qualified overload of state(), preserving read-only access.
    auto it = _states.find(typeid(StateT));
    return it == _states.end() ? nullptr : static_cast<const StateT*>(it->second.get());
}

template <typename T>
template <typename StateT>
bool
Fluid<T>::has_state() const noexcept {

    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::fluid::FluidState.");

    // Check whether a state of the requested type is currently installed.
    return _states.contains(typeid(StateT));
}

template <typename T>
template <typename StateT>
std::unique_ptr<StateT>
Fluid<T>::remove_state() {

    static_assert(std::is_base_of_v<FluidState, StateT>,
                  "StateT must derive from atlas::fluid::FluidState.");

    // Find the state by exact type. If it does not exist, return nullptr
    // to indicate that no state was removed.
    auto it = _states.find(typeid(StateT));
    if (it == _states.end()) {
        return nullptr;
    }

    // Release ownership from the base-class pointer, cast it back to the
    // requested derived type, and transfer ownership to the caller.
    auto state = std::unique_ptr<StateT>(static_cast<StateT*>(it->second.release()));

    // Remove the now-empty registry entry.
    _states.erase(it);

    return state;
}

template <typename T>
std::unordered_map<std::type_index, std::unique_ptr<FluidState>>&
Fluid<T>::states() noexcept {

    // Expose the full mutable state registry.
    //
    // This is primarily intended for high-level orchestration code that must
    // iterate across all installed states uniformly, such as compaction,
    // serialization, sink processing, or state migration utilities.
    return _states;
}

template <typename T>
const std::unordered_map<std::type_index, std::unique_ptr<FluidState>>&
Fluid<T>::states() const noexcept {

    // Const-qualified overload of states(), preserving read-only access
    // to the full installed state registry.
    return _states;
}

template <typename T>
size_t
Fluid<T>::buffer_size() const noexcept {

    // Return the fixed storage capacity of this fluid object.
    //
    // This is the maximum number of particle slots the fluid can manage
    // without reallocating or rebuilding its installed per-particle states.
    return _buffer_size;
}

template <typename T>
Fluid<T>
Fluid<T>::Builder::build() const {

    // Validate the builder configuration before constructing the Fluid.
    //
    // This ensures that all structural and physical invariants already hold.
    validate();

    Fluid<T> f(_buffer_size);

    // Copy particle/species material properties from the builder-owned
    // device buffer into the Fluid instance.
    f._particle_properties = DeviceBuffer<MatrialProperties<T>>(_particles.begin(), _particles.end());

    // Copy generator operators into the Fluid instance.
    f._generators = DeviceBuffer<GenerateOperator<T>>(_generators.begin(), _generators.end());

    // The sized constructor already installed default states and capacity.
    f._statistical_weight = _statistical_weight;
    f._observer           = _observer;

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

    // Build a temporary Fluid value first, then move it into a host-managed
    // shared allocation.
    //
    // This is a convenience API for callers who want shared ownership directly
    // from the builder.
    auto f = build();
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_properties(const HostBuffer<MatrialProperties<T>>& properties) {

    // Upload particle/species property data from host memory into the builder's
    // device buffer so the final Fluid can be constructed with device-resident
    // property storage.
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
    // the resulting buffer length remains aligned with the property table.
    _generators.clear();

    const int n = static_cast<int>(generators.size());
    for (int i = 0; i < n; ++i) {
        _generators.push_back(
            generators[i] ? generators[i]->make_generate_operator() : GenerateOperator<T> {});
    }

    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_buffer_size(const size_t buffer_size) noexcept {

    // Set the maximum number of particle slots the final Fluid instance can hold.
    //
    // This determines the size of the default per-particle states allocated
    // during Fluid(size_t) construction.
    _buffer_size = buffer_size;
    return *this;
}

template <typename T>
typename Fluid<T>::Builder&
Fluid<T>::Builder::with_statistical_weight(const T statistical_weight) noexcept {

    // Store the fluid-level statistical weight to be installed during build().
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
    const auto snapshot = atlas::serialization::load_fluid_binary<T>(path);

    _buffer_size = snapshot.buffer_size;
    _particle_count = snapshot.particle_count;
    _statistical_weight = snapshot.statistical_weight;
    _particles = DeviceBuffer<MatrialProperties<T>>(snapshot.properties.begin(), snapshot.properties.end());
    _generators = DeviceBuffer<GenerateOperator<T>>(snapshot.generators.begin(), snapshot.generators.end());

    _position_state.reset();
    _velocity_state.reset();
    _species_state.reset();
    _active_state.reset();
    _temperature_state.reset();

    if (snapshot.positions.has_value()) {
        _position_state = DeviceBuffer<Vector3<T>>(snapshot.positions->begin(), snapshot.positions->end());
    }

    if (snapshot.velocities.has_value()) {
        _velocity_state = DeviceBuffer<Vector3<T>>(snapshot.velocities->begin(), snapshot.velocities->end());
    }

    if (snapshot.species.has_value()) {
        _species_state = DeviceBuffer<std::size_t>(snapshot.species->begin(), snapshot.species->end());
    }

    if (snapshot.active.has_value()) {
        _active_state = DeviceBuffer<int>(snapshot.active->begin(), snapshot.active->end());
    }

    if (snapshot.temperature.has_value()) {
        _temperature_state = DeviceBuffer<T>(snapshot.temperature->begin(), snapshot.temperature->end());
    }

    return *this;
}

template <typename T>
void
Fluid<T>::Builder::validate() const {

    // The builder currently requires a one-to-one correspondence between
    // particle/species property entries and generator operator entries.
    //
    // A mismatch would make it ambiguous which generator belongs to which
    // property/species descriptor.
    if (_particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/generators size mismatch.");
    }

    // Statistical weight must be strictly positive.
    if (!(_statistical_weight > T(0))) {
        throw std::runtime_error(
            "Fluid::Builder: statistical_weight must be positive.");
    }

    // Validate each installed material/property entry.
    for (std::size_t i = 0; i < _particles.size(); ++i) {
        const MatrialProperties<T> particle_property = _particles[i];

        // Molecular mass must be physically meaningful.
        if (!(particle_property.molecular_mass > T(0))) {
            throw std::runtime_error(
                "Fluid::Builder: molecular_mass must be positive.");
        }

        // Enforce consistency between particle mass and the configured
        // statistical weight convention used by the simulation model.
        if (particle_property.mass != particle_property.molecular_mass * _statistical_weight) {
            throw std::runtime_error(
                "Fluid::Builder: particle mass does not match statistical weight.");
        }
    }
}

} // namespace atlas::fluid
