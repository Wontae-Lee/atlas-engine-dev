#pragma once

/**
 * @file collider.h
 * @brief Declares collider aggregation for particle-surface interactions.
 *
 * A Collider owns one or more units together with their surface interaction
 * models. It updates moving collision geometry over time and applies the
 * appropriate boundary response to particles during advection/collision stages.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Aggregates collision geometry and surface response models.
 *
 * Collider stores device-resident units and matching ColliderSurfaceInteraction
 * objects. During a collision step it advances unit poses, traces particle
 * trajectories against all registered units, and updates particle position and
 * velocity according to the selected surface interaction.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Collider final {
    static_assert(std::is_floating_point_v<T>, "Collider requires a floating-point T");

public:
    /**
     * @brief Fluent builder for Collider.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    Collider()  = default;
    /**
     * @brief Destructor.
     */
    ~Collider() = default;

    /**
     * @brief Constructs a collider from device-resident units and interactions.
     *
     * @param units Device buffer containing collision units.
     * @param surface_interactions Device buffer containing surface response models.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Collider(DeviceBuffer<Unit<T>> units,
             DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept;

    /**
     * @brief Returns a builder initialized with default values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Replaces collider units with a device buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(DeviceBuffer<Unit<T>> units) noexcept;

    /**
     * @brief Replaces collider units with host-provided values.
     *
     * @throws std::runtime_error if @p units is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Replaces surface interactions with a device buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_surface_interactions(DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept;

    /**
     * @brief Replaces surface interactions with host-provided values.
     *
     * @throws std::runtime_error if @p surface_interactions is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    /**
     * @brief Returns mutable access to collider units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Unit<T>>&
    units() noexcept;

    /**
     * @brief Returns const access to collider units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    /**
     * @brief Returns mutable access to surface interactions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<ColliderSurfaceInteraction<T>>&
    surface_interactions() noexcept;

    /**
     * @brief Returns const access to surface interactions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<ColliderSurfaceInteraction<T>>&
    surface_interactions() const noexcept;

    /**
     * @brief Advances all collider units by one timestep.
     *
     * @param dt Simulation timestep.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Resolves particle collisions against registered units.
     *
     * @param particle_probe Mutable particle probe to update in-place.
     * @param dt Simulation timestep used to build the swept particle segment.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide(FluidDeviceProbe<T>& particle_probe, T dt) const;

    /**
     * @brief Returns whether collision processing has any usable data.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    DeviceBuffer<Unit<T>> _units; ///< Collision geometry instances.
    DeviceBuffer<ColliderSurfaceInteraction<T>> _surface_interactions; ///< Surface response models.
};

template <typename T>
class Collider<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the host-side unit list.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Sets the host-side surface interaction list.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    /**
     * @brief Builds a value instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Collider<T>
    build();

    /**
     * @brief Builds a host-shared collider instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Collider<T>>
    make_host_shared();

private:
    /**
     * @brief Validates builder state before construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<Unit<T>> _units; ///< Pending host-side units.
    HostBuffer<ColliderSurfaceInteraction<T>> _surface_interactions; ///< Pending host-side interactions.
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::Collider.
 */
template <typename T>
using Collider = atlas::system::Collider<T>;

/**
 * @brief Host shared pointer alias for Collider.
 */
template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

/**
 * @brief Device shared pointer alias for Collider.
 */
template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

}

#include <atlas/collider/collider.hpp>
