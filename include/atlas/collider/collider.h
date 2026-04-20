#pragma once

/**
 * @file collider.h
 * @brief Declares the Collider class used to resolve particle collisions
 *        against time-varying collider units.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider_surface_interaction.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Resolves particle collisions between a fluid and a collection of collider units.
 *
 * A Collider owns three essential resources:
 * - a device buffer of collider units,
 * - a device buffer of surface interaction models,
 * - a host-side fluid object whose particles are tested and updated.
 *
 * Each collider unit typically combines:
 * - a geometry operator used to query intersections,
 * - a sync operator used to transform between local and world spaces,
 * - optional kinematic state that can move or rotate the collider over time.
 *
 * During collision processing for a time step @p dt:
 * 1. each particle motion is approximated by a world-space segment
 *    from its current position to position + velocity * dt,
 * 2. this motion segment is tested against every collider unit,
 * 3. the closest valid hit is selected,
 * 4. the hit point and normal are transformed back to world space,
 * 5. the particle is slightly offset along the normal to reduce re-penetration,
 * 6. a surface interaction model computes the post-collision velocity.
 *
 * Surface interaction models may be configured in two ways:
 * - one shared model for all collider units,
 * - one model per collider unit.
 *
 * @tparam T Floating-point scalar type used for all geometric and physical quantities.
 */
template <typename T>
class Collider final {
    static_assert(std::is_floating_point_v<T>, "Collider requires a floating-point T");

public:
    /**
     * @brief Builder used for validated host-side Collider construction.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty collider with no units, no fluid, and no surface interactions.
     * Such an object is not useful until properly initialized.
     */
    Collider() = default;

    /**
     * @brief Destructor.
     */
    ~Collider() = default;

    /**
     * @brief Constructs a collider from prepared units, interaction models, and a target fluid.
     *
     * This constructor assumes the provided buffers and pointers already represent
     * a semantically valid collider configuration.
     *
     * @param units Device buffer containing collider units.
     * @param surface_interactions Device buffer containing post-collision surface interaction models.
     * @param flips Device buffer storing whether each collider unit should use flipped collision normals.
     * @param fluid Host shared pointer to the target fluid whose particles will be processed.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Collider(DeviceBuffer<Unit<T>> units,
             DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions,
             DeviceBuffer<std::uint8_t> flips,
             atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Creates a fluent Builder instance for Collider construction.
     *
     * @return Newly created builder object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Advances collider unit state and resolves particle collisions for one time step.
     *
     * The default update flow is:
     * 1. advance all collider units using Unit::update(dt),
     * 2. resolve collisions between the updated units and the target fluid.
     *
     * @param dt Positive simulation time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Resolves collisions between the configured collider units and the target fluid.
     *
     * For each particle, the function builds a world-space motion segment based on
     * the particle's current velocity and the provided time step. The segment is then
     * tested against all collider units. The closest valid hit is selected and used
     * to update particle position and velocity.
     *
     * This function does not advance collider unit motion by itself; it only performs
     * collision resolution against the units' current states.
     *
     * @param dt Positive simulation time step used to define the particle motion segment.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide(T dt) const;

    /**
     * @brief Returns whether this collider lacks the minimum required configuration.
     *
     * A collider is considered effectively empty if any of the following are true:
     * - no collider units are present,
     * - no surface interaction model is available,
     * - no target fluid is attached.
     *
     * @return True if collision processing cannot be meaningfully performed.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    /**
     * @brief Device buffer containing collider units.
     *
     * Each unit provides geometry, transform information, and optional kinematic behavior.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Host-side target fluid whose particle states are modified by collision processing.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Device buffer containing surface interaction models.
     *
     * The allowed layouts are:
     * - size == 1: one shared model for all collider units,
     * - size == number of units: one interaction model per unit.
     */
    DeviceBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;

    /**
     * @brief Device buffer storing per-unit normal-flip flags.
     *
     * Each entry uses:
     * - 0: use the geometry's outward normal as-is,
     * - non-zero: invert the collision normal before resolving the response.
     *
     * Supported layouts are:
     * - size == 1: one shared flag for all units,
     * - size == number of units: one flag per unit.
     */
    DeviceBuffer<std::uint8_t> _flips;
};

/**
 * @brief Builder for validated Collider construction.
 *
 * This builder collects host-side collider units, a target fluid, and optional
 * surface interaction models before producing a device-backed Collider instance.
 *
 * Validation rules:
 * - a valid fluid must be provided,
 * - at least one collider unit must be provided,
 * - the number of surface interaction models must be:
 *   - zero, meaning a default one will be inserted during build(),
 *   - one, meaning shared across all units,
 *   - or exactly equal to the number of units.
 *
 * @tparam T Floating-point scalar type used by the collider.
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
     * The supplied units are stored in host memory inside the builder and later copied
     * into a device buffer during build().
     *
     * @param units Host buffer containing collider units.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p units is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Sets the target fluid whose particles will be collision-processed.
     *
     * @param fluid Host shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Sets the surface interaction models.
     *
     * Valid counts are:
     * - 1: shared by all units,
     * - number of units: one interaction per collider unit.
     *
     * An empty interaction set is not accepted here; omitting this call entirely
     * is the intended way to request automatic insertion of a default interaction.
     *
     * @param surface_interactions Host buffer containing surface interaction models.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p surface_interactions is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    /**
     * @brief Sets one shared flip flag for all collider units.
     *
     * @param flip Whether collision normals should be inverted before response.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    /**
     * @brief Sets per-unit normal-flip flags.
     *
     * Valid counts are:
     * - 1: shared by all units,
     * - number of units: one flag per collider unit.
     *
     * Each value is interpreted as:
     * - 0: use outward normals,
     * - non-zero: use inward/flipped normals.
     *
     * @param flips Host buffer containing flip flags.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p flips is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flips(const HostBuffer<std::uint8_t>& flips);

    /**
     * @brief Validates the configuration and builds a Collider value object.
     *
     * If no surface interaction model was explicitly provided, a single default
     * interaction model is inserted automatically.
     *
     * @return Constructed Collider object.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Collider<T>
    build();

    /**
     * @brief Builds a Collider and stores it in host-managed shared memory.
     *
     * @return Host shared pointer to the constructed collider.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Collider<T>>
    make_host_shared();

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if:
     * - no fluid is set,
     * - no units are set,
     * - the number of surface interaction models is neither 0, 1, nor equal to unit count,
     * - the number of flip flags is neither 0, 1, nor equal to unit count.
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

    /**
     * @brief Host-side normal-flip flags collected by the builder.
     */
    HostBuffer<std::uint8_t> _flips;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::Collider.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Collider = atlas::system::Collider<T>;

/**
 * @brief Host shared pointer alias for Collider.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

/**
 * @brief Device shared pointer alias for Collider.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

} // namespace atlas

#include <atlas/collider/collider.hpp>
