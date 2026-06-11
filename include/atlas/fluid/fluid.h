#pragma once

/**
 * @file fluid.h
 * @brief Defines the Fluid class and its Builder for particle-based fluid simulation.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/material/material_properties.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>

namespace atlas::fluid {

/**
 * @brief Owns particle data, generator operators, and registered simulation states
 *        for a particle-based fluid model.
 *
 * A Fluid object serves as the central runtime container for particle-oriented
 * simulation data. It stores:
 * - a device-resident table of particle/species material properties,
 * - a device-resident table of generation operators,
 * - the current number of active particles,
 * - the total particle storage capacity,
 * - a fluid-level statistical weight,
 * - a registry of installed simulation states indexed by concrete type.
 *
 * The class distinguishes clearly between:
 * - @ref buffer_size, which is the total allocated particle capacity,
 * - @ref particle_count, which is the number of logically active particles.
 *
 * The sized constructor pre-installs the default state set required by the
 * core fluid workflow:
 * - position state,
 * - velocity state,
 * - species state,
 * - active state.
 *
 * Additional states may be installed dynamically through the generic
 * state-registry API.
 *
 * @tparam T Floating-point numeric type used for simulation quantities.
 */
template <typename T>
class Fluid final {
public:
    /**
     * @brief Builder used for validated Fluid construction.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * Constructs an empty Fluid object with zero capacity, zero active particles,
     * and no pre-installed state storage.
     */
    Fluid() = default;

    /**
     * @brief Constructs a Fluid with the given particle buffer capacity.
     *
     * The constructor allocates and installs the default per-particle states
     * required by the fluid system, sized to the full buffer capacity.
     *
     * @param buffer_size Maximum number of particle slots.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Fluid(size_t buffer_size);

    /**
     * @brief Copy construction is disabled.
     */
    Fluid(const Fluid&) = delete;

    /**
     * @brief Move constructor.
     */
    Fluid(Fluid&&) noexcept = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return New builder object for fluent Fluid configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~Fluid() = default;

    /**
     * @brief Copy assignment is disabled.
     */
    Fluid&
    operator=(const Fluid&)
        = delete;

    /**
     * @brief Move assignment operator.
     */
    Fluid&
    operator=(Fluid&&) noexcept = default;

    /**
     * @brief Returns the generator operator buffer as a const reference.
     *
     * @return Const reference to the device buffer of generator operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    /**
     * @brief Returns the generator operator buffer as a mutable reference.
     *
     * @return Mutable reference to the device buffer of generator operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

    /**
     * @brief Returns the particle/species material property buffer as a const reference.
     *
     * @return Const reference to the device buffer of material properties.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<MaterialProperties<T>>&
    particle_properties() const noexcept;

    /**
     * @brief Returns the particle/species material property buffer as a mutable reference.
     *
     * @return Mutable reference to the device buffer of material properties.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<MaterialProperties<T>>&
    particle_properties() noexcept;

    /**
     * @brief Sets the number of logically active particles.
     *
     * The active particles are assumed to occupy the dense prefix
     * [0, particle_count) across all installed per-particle states.
     *
     * @param particle_count Number of active particles.
     *
     * @throw std::out_of_range Thrown if @p particle_count exceeds @ref buffer_size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_particle_count(size_t particle_count);

    /**
     * @brief Constructs and installs a state of the requested type.
     *
     * If a state of the same type already exists, it is replaced.
     *
     * @tparam StateT Concrete state type to create.
     * @tparam Args Constructor argument types.
     * @param args Constructor arguments forwarded to @p StateT.
     * @return Reference to the newly stored state.
     */
    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args);

    /**
     * @brief Installs or replaces a state via unique ownership transfer.
     *
     * @tparam StateT Concrete state type to install.
     * @param state Unique pointer owning the state object.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state);

    /**
     * @brief Retrieves a mutable pointer to the requested state type.
     *
     * @tparam StateT Concrete state type to query.
     * @return Mutable pointer to the state if present, otherwise nullptr.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE StateT*
    state() noexcept;

    /**
     * @brief Retrieves a const pointer to the requested state type.
     *
     * @tparam StateT Concrete state type to query.
     * @return Const pointer to the state if present, otherwise nullptr.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const StateT*
    state() const noexcept;

    /**
     * @brief Checks whether a state of the requested type is installed.
     *
     * @tparam StateT Concrete state type to query.
     * @return True if the state exists, false otherwise.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept;

    /**
     * @brief Removes and returns a state of the requested type.
     *
     * Ownership of the removed state is transferred to the caller.
     *
     * @tparam StateT Concrete state type to remove.
     * @return Unique pointer to the removed state, or nullptr if absent.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state();

    /**
     * @brief Returns the full state registry as a mutable reference.
     *
     * This accessor exists for orchestrating components that need to iterate
     * across every installed state uniformly, such as sinks, compactors,
     * serializers, or state-management utilities.
     *
     * @return Mutable reference to the state registry.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unordered_map<std::type_index, std::unique_ptr<FluidState>>&
    states() noexcept;

    /**
     * @brief Returns the full state registry as a const reference.
     *
     * @return Const reference to the state registry.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::unordered_map<std::type_index, std::unique_ptr<FluidState>>&
    states() const noexcept;

    /**
     * @brief Returns the total particle buffer capacity.
     *
     * @return Maximum number of particle slots.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

    /**
     * @brief Returns the current number of logically active particles.
     *
     * @return Number of active particles.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    particle_count() const noexcept;

    /**
     * @brief Returns the fluid-level statistical weight.
     *
     * @return Statistical weight.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    statistical_weight() const noexcept;

    /**
     * @brief Returns the optional observer used to record runtime metrics.
     *
     * @return Const reference to the observer shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const ObserverHostPtr&
    observer() const noexcept;

    /**
     * @brief Saves the current fluid snapshot to a protobuf-backed binary file.
     *
     * @param path Destination file path.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    save(std::string_view path) const;

private:
    friend class Builder;

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static const std::type_index&
    state_key() noexcept;

    /**
     * @brief Device buffer storing particle/species material properties.
     */
    DeviceBuffer<MaterialProperties<T>> _particle_properties;

    /**
     * @brief Device buffer storing generator operators.
     */
    DeviceBuffer<GenerateOperator<T>> _generators;

    /**
     * @brief Current number of logically active particles.
     */
    size_t _particle_count = 0;

    /**
     * @brief Maximum particle storage capacity.
     */
    size_t _buffer_size = 0;

    /**
     * @brief Statistical weight shared across this fluid.
     */
    T _statistical_weight = T(1);

    /**
     * @brief Optional observer used to record runtime metrics.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Registry of installed simulation states keyed by exact concrete type.
     */
    std::unordered_map<std::type_index, std::unique_ptr<FluidState>> _states;
};

/**
 * @brief Builder for constructing Fluid objects.
 *
 * The builder collects:
 * - particle/species material properties,
 * - generator operators,
 * - target particle buffer size,
 * - fluid-level statistical weight.
 *
 * Validation currently enforces:
 * - property count must match generator count,
 * - statistical weight must be strictly positive,
 * - molecular mass must be positive for every property entry,
 * - particle mass must match the configured statistical weight convention.
 *
 * @tparam T Floating-point numeric type used for simulation quantities.
 */
template <typename T>
class Fluid<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Validates the current configuration and builds a Fluid object.
     *
     * @return Constructed Fluid object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    /**
     * @brief Builds a Fluid object and wraps it in host-managed shared storage.
     *
     * @return Host shared pointer to the constructed Fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    /**
     * @brief Installs particle/species material properties.
     *
     * @param properties Host buffer containing material properties.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_properties(const HostBuffer<MaterialProperties<T>>& properties);

    /**
     * @brief Installs generator definitions by converting host-side generator objects
     *        into device-executable operators.
     *
     * @param generators Host buffer containing generator shared pointers.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators);

    /**
     * @brief Sets the particle buffer capacity of the final Fluid object.
     *
     * @param buffer_size Maximum number of particle slots.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    /**
     * @brief Sets the fluid-level statistical weight.
     *
     * @param statistical_weight Statistical weight value.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_statistical_weight(T statistical_weight) noexcept;

    /**
     * @brief Sets the optional observer used to record runtime metrics.
     *
     * @param observer Host shared pointer to the observer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /**
     * @brief Loads a protobuf-backed fluid snapshot into the builder.
     *
     * @param path Snapshot file path.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_binary(const std::string& path);

private:
    /**
     * @brief Validates the current builder configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Builder-owned host staging buffer for material properties.
     */
    HostBuffer<MaterialProperties<T>> _particles;

    /**
     * @brief Builder-owned host staging buffer for generator operators.
     */
    HostBuffer<GenerateOperator<T>> _generators;

    /**
     * @brief Requested particle buffer capacity.
     */
    size_t _buffer_size = 0;

    /**
     * @brief Requested fluid-level statistical weight.
     */
    T _statistical_weight = T(1.0);

    /**
     * @brief Optional observer installed into the built fluid.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Optional position state restored from a binary snapshot.
     */
    std::optional<HostBuffer<Vector3<T>>> _position_state;

    /**
     * @brief Optional velocity state restored from a binary snapshot.
     */
    std::optional<HostBuffer<Vector3<T>>> _velocity_state;

    /**
     * @brief Optional species state restored from a binary snapshot.
     */
    std::optional<HostBuffer<std::size_t>> _species_state;

    /**
     * @brief Optional active-mask state restored from a binary snapshot.
     */
    std::optional<HostBuffer<int>> _active_state;

    /**
     * @brief Optional temperature state restored from a binary snapshot.
     */
    std::optional<HostBuffer<T>> _temperature_state;

    /**
     * @brief Optional active particle count restored from a binary snapshot.
     */
    std::optional<std::size_t> _particle_count;
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for atlas::fluid::Fluid.
 *
 * @tparam T Floating-point numeric type.
 */
template <typename T>
using Fluid = fluid::Fluid<T>;

/**
 * @brief Host-side shared pointer alias for Fluid.
 *
 * @tparam T Floating-point numeric type.
 */
template <typename T>
using FluidHostPtr = host_shared_ptr<fluid::Fluid<T>>;

/**
 * @brief Device-side shared pointer alias for Fluid.
 *
 * @tparam T Floating-point numeric type.
 */
template <typename T>
using FluidDevicePtr = device_shared_ptr<fluid::Fluid<T>>;

}

#include <atlas/fluid/fluid.hpp>
