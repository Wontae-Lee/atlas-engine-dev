#pragma once

/**
 * @file despawn_operator.h
 * @brief Declares despawn operator policies used to decide whether particles
 *        should be removed relative to queried geometry.
 */

#include <atlas/sink/surface_despawn_operator.h>
#include <atlas/sink/tracing_despawn_operator.h>
#include <atlas/sink/volume_despawn_operator.h>

namespace atlas::fluid {

/**
 * @brief Runtime tag identifying which despawn rule is active.
 *
 * This tag is used by @ref DespawnOperator to dispatch at runtime between
 * the available stateless despawn policies.
 */
enum class DespawnType : int {

    /**
     * @brief Remove particles that lie on the queried geometry surface.
     */
    Surface,

    /**
     * @brief Remove particles that lie inside the queried geometry region or volume.
     */
    Volume,

    /**
     * @brief Remove particles whose velocity trace intersects the queried geometry.
     */
    Tracing
};

/**
 * @brief Runtime-dispatched despawn operator that selects between stateless policies.
 *
 * This type stores only a runtime tag describing which despawn rule is active.
 * Actual despawn logic is delegated to one of the stateless concrete policies:
 * - @ref SurfaceDespawnOperator
 * - @ref VolumeDespawnOperator
 * - @ref TracingDespawnOperator
 *
 * Because the concrete policies are stateless, this wrapper only needs to store
 * the runtime tag and does not need additional payload data.
 *
 * @tparam T Floating-point scalar type used for geometry queries.
 */
template <typename T>
struct DespawnOperator final {

    /**
     * @brief Runtime tag indicating which despawn policy is active.
     */
    DespawnType type = DespawnType::Surface;

    /**
     * @brief Default constructor.
     *
     * Initializes the operator with the default surface-based despawn policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator() noexcept = default;

    /**
     * @brief Constructs a despawn operator with the requested runtime policy type.
     *
     * @param type Runtime despawn policy tag to activate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DespawnOperator(DespawnType type) noexcept;

    /**
     * @brief Copy constructor.
     *
     * Copies the runtime policy tag from another operator.
     *
     * @param other Source operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator(const DespawnOperator& other) noexcept = default;

    /**
     * @brief Copy assignment operator.
     *
     * Replaces the current runtime policy tag with the one stored in @p other.
     *
     * @param other Source operator.
     * @return Reference to this operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DespawnOperator&
    operator=(const DespawnOperator& other) noexcept = default;

    /**
     * @brief Destructor.
     *
     * Uses the default trivial destruction behavior.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DespawnOperator() noexcept = default;

    /**
     * @brief Constructs the runtime operator from a surface despawn policy.
     *
     * Since the policy is stateless, constructing from it simply selects
     * the @ref DespawnType::Surface runtime tag.
     *
     * @param op Concrete surface despawn policy.
     */
    ATLAS_HOST
    DespawnOperator(const SurfaceDespawnOperator<T>& op);

    /**
     * @brief Constructs the runtime operator from a volume despawn policy.
     *
     * Since the policy is stateless, constructing from it simply selects
     * the @ref DespawnType::Volume runtime tag.
     *
     * @param op Concrete volume despawn policy.
     */
    ATLAS_HOST
    DespawnOperator(const VolumeDespawnOperator<T>& op);

    /**
     * @brief Constructs the runtime operator from a tracing despawn policy.
     *
     * Since the policy is stateless, constructing from it simply selects
     * the @ref DespawnType::Tracing runtime tag.
     *
     * @param op Concrete tracing despawn policy.
     */
    ATLAS_HOST
    DespawnOperator(const TracingDespawnOperator<T>& op);

    /**
     * @brief Evaluates whether a particle should be removed.
     *
     * This function dispatches the query to the currently active concrete
     * despawn policy based on the stored runtime tag.
     *
     * @param query Geometry query operator.
     * @param vector Vector passed to the selected despawn policy.
     * @param value Scalar value passed to the selected despawn policy.
     * @return True if the particle should be removed.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const atlas::geometry::GeometryOperator<T>& query,
            const Vector3<T>& vector,
            T value = T(0)) const noexcept;

    /**
     * @brief Evaluates whether a particle should be removed with explicit position.
     *
     * Surface and volume despawn policies use @p vector as their query point.
     * Tracing despawn uses @p position as the ray origin, @p vector as velocity,
     * and @p value as the time interval.
     *
     * @param query Geometry query operator.
     * @param position Particle position passed to tracing despawn.
     * @param vector Vector passed to the selected despawn policy.
     * @param value Scalar value passed to the selected despawn policy.
     * @return True if the particle should be removed.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const atlas::geometry::GeometryOperator<T>& query,
            const Vector3<T>& position,
            const Vector3<T>& vector,
            T value = T(0)) const noexcept;
};

} // namespace atlas::fluid

#include <atlas/sink/despawn_operator.hpp>
