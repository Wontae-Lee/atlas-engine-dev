#pragma once

/**
 * @file fluid_state.h
 * @brief Declares polymorphic fluid state types for per-particle device-side attributes.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>

#include <any>
#include <cstddef>

namespace atlas::fluid {

/**
 * @brief Abstract interface for a per-particle fluid state buffer.
 *
 * A FluidState models one attribute stream associated with fluid particles,
 * such as position, velocity, species, activity, or temperature.
 *
 * Each concrete implementation owns a typed device buffer and participates in
 * particle compaction through a common interface. During compaction, surviving
 * particles are gathered into a dense prefix so that all registered state
 * buffers remain index-aligned.
 */
class FluidState {
public:
    /**
     * @brief Default constructor.
     */
    FluidState() = default;

    /**
     * @brief Copy construction is disabled.
     *
     * Fluid states typically own device-resident resources and are expected to
     * be managed through unique polymorphic ownership.
     */
    FluidState(const FluidState&) = delete;

    /**
     * @brief Move constructor.
     */
    FluidState(FluidState&&) noexcept = default;

    /**
     * @brief Virtual destructor.
     */
    virtual ~FluidState() = default;

    /**
     * @brief Copy assignment is disabled.
     *
     * @return Reference to this object.
     */
    FluidState&
    operator=(const FluidState&)
        = delete;

    /**
     * @brief Move assignment operator.
     *
     * @return Reference to this object.
     */
    FluidState&
    operator=(FluidState&&) noexcept = default;

    /**
     * @brief Returns the physical size of the underlying state buffer.
     *
     * This value usually corresponds to the allocated per-particle capacity of
     * the state rather than the current number of logically active particles.
     *
     * @return Number of elements stored in the underlying buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD virtual std::size_t
    size() const noexcept = 0;

    /**
     * @brief Compacts the state buffer using a destination-to-source index map.
     *
     * The @p compact_indices buffer is expected to satisfy:
     *
     * @code
     * compact_indices[dst] = src
     * @endcode
     *
     * for each surviving destination index in the compacted prefix. Concrete
     * states typically forward this operation to compact_buffer() using their
     * internal typed buffer.
     *
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving particles to preserve in the compacted prefix.
     */
    ATLAS_HOST virtual void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept)
        = 0;

    /**
     * @brief Resets all stored particle entries to their default value.
     */
    ATLAS_HOST virtual void
    reset() = 0;

public:
    /**
     * @brief Compacts a typed state buffer into its dense surviving prefix.
     *
     * This helper gathers the first @p kept surviving elements according to
     * @p compact_indices into temporary storage and then writes the compacted
     * prefix back to @p buffer.
     *
     * The implementation is shared by concrete fluid states so they can reuse
     * the same compaction logic independent of their stored value type.
     *
     * @tparam Buffer Buffer type to compact.
     * @param buffer Buffer to compact in place.
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving elements.
     */
    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    compact_buffer(Buffer& buffer,
                   const DeviceBuffer<std::size_t>& compact_indices,
                   std::size_t kept);

    template <typename Buffer>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reset_buffer(Buffer& buffer);

public:
    /**
     * @brief Type-erased reusable scratch storage for compaction.
     *
     * This member stores temporary compacted device buffers without requiring
     * the base class to know the concrete element type at compile time.
     */
    std::any _compacted;
};

/**
 * @brief Fluid state storing per-particle positions.
 *
 * Each entry stores the 3D position of one particle.
 *
 * @tparam T Scalar type used by the position vector components.
 */
template <typename T>
class FluidPositionState final : public FluidState {
public:
    /**
     * @brief Default constructor.
     */
    FluidPositionState() = default;

    /**
     * @brief Constructs a position state with the specified capacity.
     *
     * @param buffer_size Number of position entries to allocate.
     */
    ATLAS_HOST explicit FluidPositionState(std::size_t buffer_size);

    /**
     * @brief Constructs a position state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param position Device buffer containing per-particle positions.
     */
    ATLAS_HOST explicit FluidPositionState(DeviceBuffer<Vector3<T>> position) noexcept;

    /**
     * @brief Returns the size of the underlying position buffer.
     *
     * @return Number of stored position entries.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Compacts the position buffer using the given index mapping.
     *
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving particles.
     */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    /**
     * @brief Returns mutable access to the underlying position storage.
     *
     * @return Reference to the position device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying position storage.
     *
     * @return Const reference to the position device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing per-particle positions.
     */
    DeviceBuffer<Vector3<T>> _position;
};

/**
 * @brief Fluid state storing per-particle velocities.
 *
 * Each entry stores the 3D velocity of one particle.
 *
 * @tparam T Scalar type used by the velocity vector components.
 */
template <typename T>
class FluidVelocityState final : public FluidState {
public:
    /**
     * @brief Default constructor.
     */
    FluidVelocityState() = default;

    /**
     * @brief Constructs a velocity state with the specified capacity.
     *
     * @param buffer_size Number of velocity entries to allocate.
     */
    ATLAS_HOST explicit FluidVelocityState(std::size_t buffer_size);

    /**
     * @brief Constructs a velocity state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param velocity Device buffer containing per-particle velocities.
     */
    ATLAS_HOST explicit FluidVelocityState(DeviceBuffer<Vector3<T>> velocity) noexcept;

    /**
     * @brief Returns the size of the underlying velocity buffer.
     *
     * @return Number of stored velocity entries.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Compacts the velocity buffer using the given index mapping.
     *
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving particles.
     */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    /**
     * @brief Returns mutable access to the underlying velocity storage.
     *
     * @return Reference to the velocity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Vector3<T>>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying velocity storage.
     *
     * @return Const reference to the velocity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Vector3<T>>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing per-particle velocities.
     */
    DeviceBuffer<Vector3<T>> _velocity;
};

/**
 * @brief Fluid state storing per-particle species identifiers.
 *
 * Each entry typically indicates which material or species the particle belongs to.
 *
 * @tparam T Scalar type associated with the owning fluid simulation.
 */
template <typename T>
class FluidSpeciesState final : public FluidState {
public:
    /**
     * @brief Default constructor.
     */
    FluidSpeciesState() = default;

    /**
     * @brief Constructs a species state with the specified capacity.
     *
     * @param buffer_size Number of species entries to allocate.
     */
    ATLAS_HOST explicit FluidSpeciesState(std::size_t buffer_size);

    /**
     * @brief Constructs a species state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param species Device buffer containing per-particle species identifiers.
     */
    ATLAS_HOST explicit FluidSpeciesState(DeviceBuffer<std::size_t> species) noexcept;

    /**
     * @brief Returns the size of the underlying species buffer.
     *
     * @return Number of stored species entries.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Compacts the species buffer using the given index mapping.
     *
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving particles.
     */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    /**
     * @brief Returns mutable access to the underlying species storage.
     *
     * @return Reference to the species device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<std::size_t>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying species storage.
     *
     * @return Const reference to the species device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<std::size_t>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing per-particle species identifiers.
     */
    DeviceBuffer<std::size_t> _species;
};

/**
 * @brief Fluid state storing per-particle activity flags.
 *
 * Each entry indicates whether the corresponding particle slot is logically active.
 * This state is commonly used to determine which particles survive compaction.
 *
 * @tparam T Scalar type associated with the owning fluid simulation.
 */
template <typename T>
class FluidActiveState final : public FluidState {
public:
    /**
     * @brief Default constructor.
     */
    FluidActiveState() = default;

    /**
     * @brief Constructs an activity state with the specified capacity.
     *
     * @param buffer_size Number of activity entries to allocate.
     */
    ATLAS_HOST explicit FluidActiveState(std::size_t buffer_size);

    /**
     * @brief Constructs an activity state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param active Device buffer containing per-particle activity flags.
     */
    ATLAS_HOST explicit FluidActiveState(DeviceBuffer<int> active) noexcept;

    /**
     * @brief Returns the size of the underlying activity buffer.
     *
     * @return Number of stored activity entries.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Compacts the activity buffer using the given index mapping.
     *
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving particles.
     */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    /**
     * @brief Returns mutable access to the underlying activity storage.
     *
     * @return Reference to the activity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<int>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying activity storage.
     *
     * @return Const reference to the activity device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<int>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing per-particle active flags.
     */
    DeviceBuffer<int> _active;
};

/**
 * @brief Fluid state storing per-particle temperatures.
 *
 * Each entry stores the temperature associated with one particle.
 *
 * @tparam T Scalar type used to represent temperature values.
 */
template <typename T>
class FluidTemperatureState final : public FluidState {
public:
    /**
     * @brief Default constructor.
     */
    FluidTemperatureState() = default;

    /**
     * @brief Constructs a temperature state with the specified capacity.
     *
     * @param buffer_size Number of temperature entries to allocate.
     */
    ATLAS_HOST explicit FluidTemperatureState(std::size_t buffer_size);

    /**
     * @brief Constructs a temperature state from an existing device buffer.
     *
     * Ownership of the provided buffer is transferred to this state.
     *
     * @param temperature Device buffer containing per-particle temperatures.
     */
    ATLAS_HOST explicit FluidTemperatureState(DeviceBuffer<T> temperature) noexcept;

    /**
     * @brief Returns the size of the underlying temperature buffer.
     *
     * @return Number of stored temperature entries.
     */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    size() const noexcept override;

    /**
     * @brief Compacts the temperature buffer using the given index mapping.
     *
     * @param compact_indices Device buffer mapping compacted destination indices
     *        to original source indices.
     * @param kept Number of surviving particles.
     */
    ATLAS_HOST void
    compact(const DeviceBuffer<std::size_t>& compact_indices, std::size_t kept) override;

    ATLAS_HOST void
    reset() override;

    /**
     * @brief Returns mutable access to the underlying temperature storage.
     *
     * @return Reference to the temperature device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<T>&
    data() noexcept;

    /**
     * @brief Returns read-only access to the underlying temperature storage.
     *
     * @return Const reference to the temperature device buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<T>&
    data() const noexcept;

private:
    /**
     * @brief Device buffer storing per-particle temperatures.
     */
    DeviceBuffer<T> _temperature;
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for atlas::fluid::FluidState.
 */
using FluidState = atlas::fluid::FluidState;

/**
 * @brief Alias for atlas::fluid::FluidPositionState.
 *
 * @tparam T Scalar type used by the state.
 */
template <typename T>
using FluidPositionState = atlas::fluid::FluidPositionState<T>;

/**
 * @brief Alias for atlas::fluid::FluidVelocityState.
 *
 * @tparam T Scalar type used by the state.
 */
template <typename T>
using FluidVelocityState = atlas::fluid::FluidVelocityState<T>;

/**
 * @brief Alias for atlas::fluid::FluidSpeciesState.
 *
 * @tparam T Scalar type used by the state.
 */
template <typename T>
using FluidSpeciesState = atlas::fluid::FluidSpeciesState<T>;

/**
 * @brief Alias for atlas::fluid::FluidActiveState.
 *
 * @tparam T Scalar type used by the state.
 */
template <typename T>
using FluidActiveState = atlas::fluid::FluidActiveState<T>;

/**
 * @brief Alias for atlas::fluid::FluidTemperatureState.
 *
 * @tparam T Scalar type used by the state.
 */
template <typename T>
using FluidTemperatureState = atlas::fluid::FluidTemperatureState<T>;

} // namespace atlas

#include <atlas/fluid/fluid_state.hpp>
