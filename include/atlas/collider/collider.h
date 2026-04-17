#pragma once

/**
 * @file collider.h
 * @brief Declares the Collider class used to resolve particle collisions against collider units.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <type_traits>

namespace atlas::system {

/**
 * @brief Particle collider for fluid particles and geometric units.
 *
 * A Collider stores:
 * - collider units defining collision geometry and transforms,
 * - surface interaction models describing post-collision response,
 * - a target fluid whose particles are tested against the collider geometry.
 *
 * During collision resolution, each particle is advanced over a segment defined
 * by its current velocity and the given time step. The collider finds the
 * closest valid hit among all configured units and applies the corresponding
 * surface interaction model to update particle position and velocity.
 *
 * @tparam T Floating-point scalar type used by the collider and fluid.
 */
template <typename T>
class Collider final {
    static_assert(std::is_floating_point_v<T>, "Collider requires a floating-point T");

public:
    /**
     * @brief Builder for configuring and constructing Collider instances.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    Collider() = default;

    /**
     * @brief Destructor.
     */
    ~Collider() = default;

    /**
     * @brief Constructs a collider from prepared units, interactions, and a fluid.
     *
     * @param units Device buffer containing collider units.
     * @param surface_interactions Device buffer containing surface interaction models.
     * @param fluid Host shared pointer to the target fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Collider(DeviceBuffer<Unit<T>> units,
             DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions,
             atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent Collider construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Updates all collider units with the given time step.
     *
     * This function is typically used to advance time-dependent state of the
     * collider units before collision processing.
     *
     * @param dt Time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Resolves collisions for all active particles in the target fluid.
     *
     * For each particle, the collider tests the motion segment implied by
     * velocity * dt against all configured units, chooses the closest valid hit,
     * and applies the corresponding surface interaction model.
     *
     * @param dt Time step used to define the particle motion segment.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide(T dt) const;

    /**
     * @brief Returns whether the collider is effectively empty.
     *
     * A collider is considered empty when it has no units, no valid fluid, or
     * no surface interaction configuration available.
     *
     * @return True if collision processing cannot meaningfully proceed.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    /**
     * @brief Collider units providing geometry and transform information.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Target fluid whose particles are processed by this collider.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Surface interaction models applied after collision.
     *
     * This buffer may contain either:
     * - one shared interaction used for all units, or
     * - one interaction per unit.
     */
    DeviceBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;
};

/**
 * @brief Builder for Collider.
 *
 * This builder collects collider units, the target fluid, and surface
 * interaction models before constructing a validated Collider object.
 *
 * Validation ensures that:
 * - a target fluid is provided,
 * - at least one collider unit exists,
 * - surface interaction count is either 1 or matches the unit count.
 *
 * @tparam T Floating-point scalar type used by the collider and fluid.
 */
template <typename T>
class Collider<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the collider units.
     *
     * @param units Host buffer containing collider units.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p units is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Sets the target fluid.
     *
     * @param fluid Host shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Sets the surface interaction models.
     *
     * The interaction count must be either:
     * - exactly 1, to share a single interaction across all units, or
     * - equal to the number of units.
     *
     * @param surface_interactions Host buffer containing surface interaction models.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p surface_interactions is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    /**
     * @brief Builds a validated Collider object.
     *
     * If no surface interaction models were explicitly provided, a default
     * interaction is inserted automatically.
     *
     * @return Constructed Collider object.
     *
     * @throw std::runtime_error Thrown if the builder configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Collider<T>
    build();

    /**
     * @brief Builds a host-side shared Collider object.
     *
     * @return Host shared pointer to a constructed Collider object.
     *
     * @throw std::runtime_error Thrown if the builder configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Collider<T>>
    make_host_shared();

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Host-side collider units collected by the builder.
     */
    HostBuffer<Unit<T>> _units;

    /**
     * @brief Target fluid collected by the builder.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Host-side surface interaction models collected by the builder.
     */
    HostBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::Collider.
 *
 * @tparam T Floating-point scalar type used by the collider.
 */
template <typename T>
using Collider = atlas::system::Collider<T>;

/**
 * @brief Host-side shared pointer alias for Collider.
 *
 * @tparam T Floating-point scalar type used by the collider.
 */
template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

/**
 * @brief Device-side shared pointer alias for Collider.
 *
 * @tparam T Floating-point scalar type used by the collider.
 */
template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

} // namespace atlas

#include <atlas/collider/collider.hpp>