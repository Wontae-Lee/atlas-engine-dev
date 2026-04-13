#pragma once

/**
 * @file despawn_operator.h
 * @brief Declares backend-portable particle despawn classification operators and their tagged-union wrapper.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::DespawnType, which selects the geometric despawn rule,
 * - @ref atlas::system::SurfaceDespawnOperator, which classifies particles relative
 *   to a geometry surface for removal,
 * - @ref atlas::system::VolumeDespawnOperator, which classifies particles relative
 *   to a geometry volume for removal,
 * - @ref atlas::system::DespawnOperator, a lightweight tagged-union wrapper used
 *   to erase the concrete despawn rule into a backend-portable value type.
 *
 * ## Purpose
 * Despawn operators are used by sinks to decide whether an active particle should
 * be removed with respect to a geometry instance.
 *
 * Typical uses include:
 * - removing particles that hit or lie on a surface,
 * - removing particles that enter or remain inside a volume,
 * - switching between despawn rules without changing higher-level sink logic.
 *
 * ## Design
 * The concrete despawn policies are represented as lightweight stateless operator
 * types. The public @ref DespawnOperator wrapper stores one active policy selected
 * by a runtime @ref DespawnType tag.
 *
 * This design allows:
 * - host/device portability,
 * - value-type storage in buffers,
 * - runtime dispatch without virtual inheritance.
 *
 * ## Tolerance
 * Both despawn policies accept an optional geometric tolerance parameter. This can
 * be used to:
 * - widen surface classification bands,
 * - soften strict boundary checks,
 * - improve numerical robustness near geometry boundaries.
 *
 * ## Host/device usage
 * All query-facing entry points are marked `ATLAS_ALL_DEVICE`, allowing despawn
 * classification to be used consistently in:
 * - host-side code,
 * - device kernels,
 * - backend-portable sink logic.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and tolerances.
 */

#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Selects the geometric despawn rule used by a sink.
 *
 * @details
 * This enum identifies which concrete despawn policy is active in a
 * @ref DespawnOperator.
 */
enum class DespawnType : int {
    /**
     * @brief Despawn relative to a geometry surface.
     *
     * @details
     * Candidate particles are classified according to whether they lie on or near
     * the geometry surface, subject to the supplied tolerance.
     */
    Surface,

    /**
     * @brief Despawn relative to a geometry volume.
     *
     * @details
     * Candidate particles are classified according to whether they lie inside the
     * geometry volume, subject to the supplied tolerance.
     */
    Volume
};

/**
 * @brief Surface-based despawn policy.
 *
 * @details
 * @ref SurfaceDespawnOperator classifies a particle with respect to the surface
 * of a geometry object.
 *
 * This policy is typically used when particles should be removed:
 * - directly on contact with a boundary,
 * - within a thin shell around a surface,
 * - at a geometric interface rather than throughout the interior.
 *
 * The exact classification rule is implementation-defined in
 * `despawn_operator.hpp`, but it typically relies on:
 * - signed-distance evaluation,
 * - surface classification,
 * - a tolerance band around the surface.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and tolerances.
 */
template <typename T>
struct SurfaceDespawnOperator final {
    /**
     * @brief Return whether a particle should be removed according to the surface rule.
     *
     * @details
     * Evaluates the particle position against the supplied geometry query object
     * and decides whether it belongs to the active despawn region associated
     * with the geometry surface.
     *
     * @param query Geometry operator used for surface classification.
     * @param particle Particle/sample position.
     * @param tolerance Optional geometric tolerance applied to the surface test.
     * @return `true` if the particle should be removed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

/**
 * @brief Volume-based despawn policy.
 *
 * @details
 * @ref VolumeDespawnOperator classifies a particle with respect to the interior
 * of a geometry object.
 *
 * This policy is typically used when particles should be removed:
 * - after entering a region,
 * - while residing inside a filled volume,
 * - when crossing into a forbidden domain.
 *
 * The exact classification rule is implementation-defined in
 * `despawn_operator.hpp`, but it typically relies on:
 * - inside/outside classification,
 * - signed-distance evaluation,
 * - an optional tolerance around the interior boundary.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and tolerances.
 */
template <typename T>
struct VolumeDespawnOperator final {
    /**
     * @brief Return whether a particle should be removed according to the volume rule.
     *
     * @details
     * Evaluates the particle position against the supplied geometry query object
     * and decides whether it belongs to the active despawn region associated
     * with the geometry interior.
     *
     * @param query Geometry operator used for volume classification.
     * @param particle Particle/sample position.
     * @param tolerance Optional geometric tolerance applied to the volume test.
     * @return `true` if the particle should be removed; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

/**
 * @brief Tagged-union wrapper over all supported despawn policies.
 *
 * @details
 * @ref DespawnOperator stores one active concrete despawn policy together with a
 * runtime @ref DespawnType tag identifying which policy is currently active.
 *
 * The wrapper provides:
 * - default construction,
 * - tag-based construction,
 * - copy construction and copy assignment,
 * - explicit construction from concrete policy objects,
 * - a single dispatching @ref despawn entry point.
 *
 * ## Active state
 * The active union member is determined by @ref type:
 * - `DespawnType::Surface` activates @ref surface,
 * - `DespawnType::Volume` activates @ref volume.
 *
 * ## Intended use
 * This wrapper is designed for:
 * - storing despawn rules in buffers,
 * - passing despawn policies into host/device code,
 * - runtime selection of despawn behavior without polymorphic allocation.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the despawn rules.
 */
template <typename T>
struct DespawnOperator final {
    /**
     * @brief Active despawn-policy tag.
     *
     * @details
     * Determines which union member is currently active.
     */
    DespawnType type = DespawnType::Surface;

    /**
     * @brief Union storing the active concrete despawn policy.
     *
     * @details
     * Only the member corresponding to @ref type is considered active.
     */
    union {
        /**
         * @brief Surface-based despawn policy.
         */
        SurfaceDespawnOperator<T> surface;

        /**
         * @brief Volume-based despawn policy.
         */
        VolumeDespawnOperator<T> volume;
    };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a despawn operator with the default active type and corresponding
     * default-initialized union member.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator() noexcept;

    /**
     * @brief Construct a despawn operator from a runtime despawn type.
     *
     * @details
     * Activates the union member corresponding to @p type.
     *
     * @param type Active despawn-policy tag.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DespawnOperator(DespawnType type) noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies the active tag and reconstructs the corresponding active union member.
     *
     * @param other Source despawn operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator(const DespawnOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Replaces the current active state with the active state of @p other.
     *
     * @param other Source despawn operator.
     * @return `*this`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DespawnOperator&
    operator=(const DespawnOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * @details
     * Destroys the currently active union member.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DespawnOperator() noexcept;

    /**
     * @brief Construct a despawn operator from a concrete surface policy.
     *
     * @param op Source surface despawn operator.
     */
    ATLAS_HOST
    DespawnOperator(const SurfaceDespawnOperator<T>& op);

    /**
     * @brief Construct a despawn operator from a concrete volume policy.
     *
     * @param op Source volume despawn operator.
     */
    ATLAS_HOST
    DespawnOperator(const VolumeDespawnOperator<T>& op);

    /**
     * @brief Dispatch despawn classification to the active concrete policy.
     *
     * @details
     * Forwards the query to the currently active despawn operator selected by @ref type.
     *
     * @param query Geometry operator used for classification.
     * @param particle Particle/sample position.
     * @param tolerance Optional geometric tolerance applied to the active test.
     * @return `true` if the particle satisfies the active despawn rule; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) const noexcept;

private:
    /**
     * @brief Destroy the currently active union member.
     *
     * @details
     * Uses @ref type to determine which union member is active and calls the
     * corresponding destructor.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    /**
     * @brief Copy the active state from another despawn operator.
     *
     * @details
     * Uses the source operator's @ref type to reconstruct the corresponding
     * active union member in this object.
     *
     * @param other Source despawn operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DespawnOperator& other) noexcept;
};

} // namespace atlas::system

#include <atlas/sink/despawn_operator.hpp>