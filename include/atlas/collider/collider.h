#pragma once

/**
 * @file collider.h
 * @brief Declares the collider system that aggregates collision geometry and surface interaction policies.
 *
 * @details
 * This header defines @ref atlas::system::Collider, a container-like system object
 * responsible for managing solid collision geometry together with the boundary
 * response policies applied when particles hit that geometry.
 *
 * A collider is composed of two parallel collections:
 * - a set of @ref atlas::Unit objects representing collision geometry instances,
 * - a set of @ref atlas::system::ColliderSurfaceInteraction objects representing
 *   the surface-response rule associated with each unit.
 *
 * During simulation, the collider typically participates in two stages:
 * - **update**, where collision units are advanced in time so their geometry
 *   remains synchronized with the current simulation state,
 * - **collide**, where particle trajectories are tested against the registered
 *   units and, upon contact, particle state is modified according to the matching
 *   surface interaction model.
 *
 * ## Role in the pipeline
 * The collider provides a centralized mechanism for particle-boundary handling:
 * - moving or animated collision units can be advanced every timestep,
 * - particle sweeps can be traced against all registered geometry,
 * - the appropriate surface interaction policy can be selected per hit,
 * - particle position and velocity can be updated in-place after impact.
 *
 * ## Data layout
 * The class stores its primary state in device-resident buffers so that:
 * - collision units are accessible to backend execution code,
 * - interaction policies are available during particle collision resolution,
 * - host-side setup can be transferred efficiently into simulation-ready memory.
 *
 * ## Mapping between units and interactions
 * The implementation assumes a logical correspondence between the stored units
 * and surface interaction objects. In the common case, element `i` in the unit
 * buffer is paired with element `i` in the surface interaction buffer.
 *
 * The exact validation and mismatch-handling policy is implementation-defined in
 * `collider.hpp`.
 *
 * ## Construction
 * A collider may be:
 * - default-constructed and then populated through setters, or
 * - created through the nested fluent @ref Builder, which stages host-side unit
 *   and interaction lists, validates them, and constructs the final object.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used throughout geometry, kinematics, and collision response.
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
 * @brief Aggregates collision geometry and surface-response models for particle simulation.
 *
 * @details
 * @ref Collider owns the collision-side data required to resolve particle-surface
 * interactions against one or more registered solid objects.
 *
 * Specifically, it stores:
 * - device-resident collision units describing the geometry to be tested,
 * - device-resident surface interaction policies describing how particles respond
 *   after hitting each corresponding unit.
 *
 * ## High-level behavior
 * A typical timestep involving a collider consists of:
 * 1. Calling @ref update to advance all registered units according to their own
 *    motion or synchronization logic.
 * 2. Calling @ref collide to sweep particle motion over the timestep and test
 *    for intersections against those units.
 * 3. Updating particle position and velocity in-place using the matched
 *    @ref ColliderSurfaceInteraction.
 *
 * ## Geometry and response coupling
 * Each collision unit represents the shape and pose used for hit testing, while
 * each surface interaction object represents the post-impact boundary law, such as:
 * - specular reflection,
 * - diffuse re-emission,
 * - restitution scaling,
 * - mixed accommodation behavior.
 *
 * ## Device-oriented storage
 * Both units and interaction policies are stored in @ref DeviceBuffer containers,
 * enabling efficient use in backend execution paths and simulation kernels.
 *
 * ## Validity expectations
 * Meaningful collision processing generally requires:
 * - at least one registered collision unit,
 * - at least one registered surface interaction,
 * - a consistent mapping between the two collections.
 *
 * The exact admissibility checks are implementation-defined in `collider.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Collider final {
    static_assert(std::is_floating_point_v<T>, "Collider requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Collider.
     *
     * @details
     * The builder stages host-side unit and interaction arrays, validates them,
     * and then materializes either:
     * - a value instance of @ref Collider, or
     * - a host-owned shared pointer to such an instance.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty collider with no registered units and no registered
     * surface interaction policies.
     *
     * @note
     * A default-constructed collider is typically not useful for collision
     * processing until its unit and interaction buffers are populated.
     */
    Collider() = default;

    /**
     * @brief Default destructor.
     *
     * @details
     * Since the class relies on RAII-managed member objects for its resources,
     * the destructor is defaulted.
     */
    ~Collider() = default;

    /**
     * @brief Construct a collider from device-resident units and interaction policies.
     *
     * @details
     * Initializes the collider directly from device-side buffers that are already
     * prepared for simulation use.
     *
     * This constructor is appropriate when:
     * - unit geometry has already been uploaded to device memory,
     * - surface interaction policies have already been uploaded to device memory,
     * - no intermediate host-side staging through the builder is required.
     *
     * @param units Device buffer containing collision geometry units.
     * @param surface_interactions Device buffer containing surface interaction policies.
     *
     * @note
     * Logical correspondence between units and surface interactions is expected
     * but validated according to the implementation in `collider.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Collider(DeviceBuffer<Unit<T>> units,
             DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder for fluent collider construction.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Replace the collider's registered units using a device buffer.
     *
     * @details
     * Transfers ownership or contents of the supplied device-resident unit buffer
     * into this collider.
     *
     * @param units Device buffer containing collision units.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(DeviceBuffer<Unit<T>> units) noexcept;

    /**
     * @brief Replace the collider's registered units using host-side values.
     *
     * @details
     * Accepts a host buffer of units and updates the collider's internal
     * device-resident unit storage accordingly.
     *
     * This is the convenient entry point when collision geometry is assembled
     * on the host and must be uploaded for simulation use.
     *
     * @param units Host buffer containing collision units.
     *
     * @throws std::runtime_error
     * Thrown if @p units is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Replace the collider's surface interactions using a device buffer.
     *
     * @details
     * Transfers ownership or contents of the supplied device-resident interaction
     * buffer into this collider.
     *
     * @param surface_interactions Device buffer containing surface-response models.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_surface_interactions(DeviceBuffer<ColliderSurfaceInteraction<T>> surface_interactions) noexcept;

    /**
     * @brief Replace the collider's surface interactions using host-side values.
     *
     * @details
     * Accepts a host buffer of interaction policies and updates the collider's
     * internal device-resident interaction storage accordingly.
     *
     * @param surface_interactions Host buffer containing surface-response models.
     *
     * @throws std::runtime_error
     * Thrown if @p surface_interactions is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    /**
     * @brief Return mutable access to the registered collision units.
     *
     * @details
     * Exposes the internal device buffer of collision units so callers may
     * inspect or modify the active collision geometry set.
     *
     * @return Mutable reference to the internal unit buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Unit<T>>&
    units() noexcept;

    /**
     * @brief Return const access to the registered collision units.
     *
     * @details
     * Exposes the internal device buffer of collision units for read-only access.
     *
     * @return Const reference to the internal unit buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    /**
     * @brief Return mutable access to the registered surface interaction policies.
     *
     * @details
     * Exposes the internal device buffer of interaction models so callers may
     * inspect or modify the active particle-surface response rules.
     *
     * @return Mutable reference to the internal interaction buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<ColliderSurfaceInteraction<T>>&
    surface_interactions() noexcept;

    /**
     * @brief Return const access to the registered surface interaction policies.
     *
     * @details
     * Exposes the internal device buffer of interaction models for read-only access.
     *
     * @return Const reference to the internal interaction buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<ColliderSurfaceInteraction<T>>&
    surface_interactions() const noexcept;

    /**
     * @brief Advance all registered collider units by one simulation timestep.
     *
     * @details
     * Updates the internal collision geometry so that moving or time-dependent
     * units remain synchronized with the current simulation time.
     *
     * Depending on the underlying @ref Unit implementation, this may involve:
     * - advancing rigid-body motion,
     * - updating transforms through synchronization objects,
     * - refreshing geometry state used during subsequent tracing.
     *
     * @param dt Simulation timestep.
     *
     * @note
     * This function updates collider geometry only. It does not itself resolve
     * particle collisions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Resolve particle collisions against all registered collision units.
     *
     * @details
     * Sweeps particle motion over the given timestep and tests for intersections
     * against the collider's registered units.
     *
     * On collision, the implementation typically:
     * - builds a swept particle segment from the particle's current state and @p dt,
     * - traces that motion against candidate units,
     * - determines the relevant hit location and surface normal,
     * - selects the corresponding @ref ColliderSurfaceInteraction,
     * - updates particle position and velocity in-place according to the chosen
     *   boundary response.
     *
     * @param particle_probe Mutable fluid probe whose particle state will be updated in-place.
     * @param dt Simulation timestep used to define the swept particle trajectory.
     *
     * @note
     * The exact collision order, tie-breaking behavior, and multi-hit handling
     * policy are implementation-defined in `collider.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    collide(FluidDeviceProbe<T>& particle_probe, T dt) const;

    /**
     * @brief Return whether the collider has no usable collision data.
     *
     * @details
     * This is a convenience validity check for determining whether collision
     * processing can proceed meaningfully.
     *
     * A collider is typically considered empty when it lacks:
     * - registered collision units,
     * - registered surface interactions,
     * - or both.
     *
     * The exact criterion is implementation-defined in `collider.hpp`.
     *
     * @return `true` if the collider has no usable collision content; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

private:
    /**
     * @brief Device-resident collision geometry instances.
     *
     * @details
     * Stores the collection of units used for particle hit testing.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Device-resident surface-response models.
     *
     * @details
     * Stores the collection of surface interaction policies associated with the
     * registered collision units.
     */
    DeviceBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;
};

/**
 * @brief Fluent builder for @ref Collider.
 *
 * @details
 * The builder provides a controlled construction path for @ref Collider using
 * host-side staged data.
 *
 * It allows callers to:
 * - supply a host-side list of collision units,
 * - supply a host-side list of surface interaction policies,
 * - validate that the staged state is suitable for construction,
 * - build either a value instance or a host-owned shared pointer.
 *
 * ## Typical usage
 * @code
 * auto collider = atlas::Collider<float>::builder()
 *     .with_units(units)
 *     .with_surface_interactions(interactions)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - the unit list is not empty,
 * - the interaction list is not empty,
 * - the two lists are compatible in size and mapping semantics.
 *
 * The exact validation rules are implementation-defined in `collider.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Collider<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with empty staged unit and interaction buffers.
     */
    Builder() = default;

    /**
     * @brief Set the host-side list of collision units.
     *
     * @details
     * Stores the supplied host buffer of units so it can later be validated and
     * transferred into the constructed collider.
     *
     * @param units Host buffer containing collision geometry instances.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Set the host-side list of surface interaction policies.
     *
     * @details
     * Stores the supplied host buffer of interaction policies so it can later be
     * validated and transferred into the constructed collider.
     *
     * @param surface_interactions Host buffer containing particle-surface response models.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_surface_interactions(const HostBuffer<ColliderSurfaceInteraction<T>>& surface_interactions);

    /**
     * @brief Build a configured @ref Collider by value after validation.
     *
     * @details
     * Validates the staged builder state, transfers the staged host-side data
     * into device-resident collider storage, and returns the constructed collider
     * by value.
     *
     * @return Fully constructed collider instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Collider<T>
    build();

    /**
     * @brief Build a configured @ref Collider in a host_shared_ptr after validation.
     *
     * @details
     * Validates the staged builder state, constructs the collider, and returns it
     * in a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<Collider<T>>` owning the constructed collider.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Collider<T>>
    make_host_shared();

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged unit and interaction arrays.
     *
     * Typical checks may include:
     * - the unit list is non-empty,
     * - the interaction list is non-empty,
     * - the staged collections are mutually compatible.
     *
     * @note
     * The exact validation policy is implementation-defined in `collider.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending host-side collision units.
     *
     * @details
     * Staged geometry instances that will be transferred into the final collider
     * during build.
     */
    HostBuffer<Unit<T>> _units;

    /**
     * @brief Pending host-side surface interaction policies.
     *
     * @details
     * Staged boundary-response models that will be transferred into the final
     * collider during build.
     */
    HostBuffer<ColliderSurfaceInteraction<T>> _surface_interactions;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Collider.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Collider = atlas::system::Collider<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Collider.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderHostPtr = atlas::host_shared_ptr<Collider<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::Collider.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using ColliderDevicePtr = atlas::device_shared_ptr<Collider<T>>;

} // namespace atlas

#include <atlas/collider/collider.hpp>