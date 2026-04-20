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

#include <memory>
#include <typeindex>
#include <unordered_map>

namespace atlas::fluid {

/**
 * @brief Represents a particle-based fluid simulation container.
 *
 * This class manages particle data, generators, and simulation states.
 * It uses GPU device buffers for efficient parallel processing.
 *
 * @tparam T Numeric type used for simulation (e.g., float, double).
 */
template <typename T>
class Fluid final {
public:
    /**
     * @brief Builder class for constructing Fluid instances.
     */
    class Builder;

    /**
     * @brief Default constructor.
     */
    Fluid() = default;

    /**
     * @brief Constructs a Fluid with a given buffer size.
     *
     * @param buffer_size Maximum number of particles the buffer can hold.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Fluid(size_t buffer_size);

    Fluid(const Fluid&)     = delete;
    Fluid(Fluid&&) noexcept = default;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for configuring a Fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~Fluid() = default;

    Fluid&
    operator=(const Fluid&)
        = delete;
    Fluid&
    operator=(Fluid&&) noexcept = default;

    /**
     * @brief Gets the generator buffer (const).
     *
     * @return Const reference to device buffer of generators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    /**
     * @brief Gets the generator buffer.
     *
     * @return Reference to device buffer of generators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

    /**
     * @brief Gets the particle material/property buffer (const).
     *
     * @return Const reference to device buffer of particle properties.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<MatrialProperties<T>>&
    particle_properties() const noexcept;

    /**
     * @brief Gets the particle material/property buffer.
     *
     * @return Reference to device buffer of particle properties.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<MatrialProperties<T>>&
    particle_properties() noexcept;

    /**
     * @brief Removes all particles from the fluid.
     *
     * This resets the particle count but does not necessarily deallocate buffers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    remove_particles();

    /**
     * @brief Sets the current number of active particles.
     *
     * @param particle_count Number of active particles in the dense prefix.
     *
     * @throw std::out_of_range Thrown if @p particle_count exceeds @ref buffer_size.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_particle_count(size_t particle_count);

    /**
     * @brief Constructs and inserts a new state.
     *
     * @tparam StateT Type of the state.
     * @tparam Args Constructor argument types.
     * @param args Arguments forwarded to the state constructor.
     * @return Reference to the created state.
     */
    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args);

    /**
     * @brief Sets a state using a unique pointer.
     *
     * @tparam StateT Type of the state.
     * @param state Unique pointer to the state.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state);

    /**
     * @brief Retrieves a mutable pointer to a state.
     *
     * @tparam StateT Type of the state.
     * @return Pointer to the state, or nullptr if not found.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE StateT*
    state() noexcept;

    /**
     * @brief Retrieves a const pointer to a state.
     *
     * @tparam StateT Type of the state.
     * @return Const pointer to the state, or nullptr if not found.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const StateT*
    state() const noexcept;

    /**
     * @brief Checks if a state exists.
     *
     * @tparam StateT Type of the state.
     * @return True if the state exists, false otherwise.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept;

    /**
     * @brief Removes and returns a state.
     *
     * @tparam StateT Type of the state.
     * @return Unique pointer to the removed state.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state();

    /**
     * @brief Gets the buffer size.
     *
     * @return Maximum number of particles.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

    /**
     * @brief Gets the current particle count.
     *
     * @return Number of active particles.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    particle_count() const noexcept;

    /**
     * @brief Gets the fluid-level statistical weight.
     *
     * @return Statistical weight.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    statistical_weight() const noexcept;

private:
    friend class Builder;

    /**
     * @brief Device buffer storing particle material properties.
     */
    DeviceBuffer<MatrialProperties<T>> _particle_properties;

    /**
     * @brief Device buffer storing particle generators.
     */
    DeviceBuffer<GenerateOperator<T>> _generators;

    /**
     * @brief Buffer used to mark particles to keep during compaction.
     */
    DeviceBuffer<std::size_t> _keep;

    /**
     * @brief Prefix sum offsets used in compaction.
     */
    DeviceBuffer<std::size_t> _offsets;

    /**
     * @brief Buffer storing compacted particle indices.
     */
    DeviceBuffer<std::size_t> _compact_indices;

    /**
     * @brief Current number of active particles.
     */
    size_t _particle_count = 0;

    /**
     * @brief Maximum capacity of the particle buffer.
     */
    size_t _buffer_size = 0;

    /**
     * @brief Statistical weight shared across this fluid.
     */
    T _statistical_weight = T(1);

    /**
     * @brief Map storing simulation states indexed by type.
     */
    std::unordered_map<std::type_index, std::unique_ptr<FluidState>> _states;
};

/**
 * @brief Builder for constructing Fluid objects.
 *
 * Provides a fluent interface for configuring Fluid instances.
 *
 * @tparam T Numeric type used for simulation.
 */
template <typename T>
class Fluid<T>::Builder final {
public:
    Builder() = default;

    /**
     * @brief Builds a Fluid instance.
     *
     * @return Constructed Fluid object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    /**
     * @brief Builds a shared pointer to a Fluid instance.
     *
     * @return Shared pointer to Fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    /**
     * @brief Sets particle properties.
     *
     * @param properties Host buffer containing particle properties.
     * @return Reference to this Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_properties(const HostBuffer<MatrialProperties<T>>& properties);

    /**
     * @brief Sets generators.
     *
     * @param generators Host buffer of generator pointers.
     * @return Reference to this Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_generators(const HostBuffer<GeneratorHostPtr<T>>& generators);

    /**
     * @brief Sets buffer size.
     *
     * @param buffer_size Maximum number of particles.
     * @return Reference to this Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    /**
     * @brief Sets the fluid-level statistical weight.
     *
     * @param statistical_weight Statistical weight value.
     * @return Reference to this Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_statistical_weight(T statistical_weight) noexcept;

private:
    /**
     * @brief Validates builder configuration.
     *
     * Ensures required parameters are set before building.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DeviceBuffer<MatrialProperties<T>> _particles;
    DeviceBuffer<GenerateOperator<T>> _generators;
    size_t _buffer_size = 0;
    T _statistical_weight = T(1.0);
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for Fluid.
 */
template <typename T>
using Fluid = fluid::Fluid<T>;

/**
 * @brief Host-side shared pointer to Fluid.
 */
template <typename T>
using FluidHostPtr = host_shared_ptr<fluid::Fluid<T>>;

/**
 * @brief Device-side shared pointer to Fluid.
 */
template <typename T>
using FluidDevicePtr = device_shared_ptr<fluid::Fluid<T>>;

}

#include <atlas/fluid/fluid.hpp>
