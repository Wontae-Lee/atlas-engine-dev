#pragma once

/**
 * @file spawn_operator.h
 * @brief Declares backend-portable particle spawn classification operators and their tagged-union wrapper.
 *
 * @details
 * This header defines:
 * - @ref atlas::system::SpawnType, which selects the geometric spawning rule,
 * - @ref atlas::system::SurfaceSpawnOperator, which classifies particles relative
 *   to a geometry surface,
 * - @ref atlas::system::VolumeSpawnOperator, which classifies particles relative
 *   to a geometry volume,
 * - @ref atlas::system::SpawnOperator, a lightweight tagged-union wrapper used
 *   to erase the concrete spawn rule into a backend-portable value type.
 *
 * ## Purpose
 * Spawn operators are used by particle sources to decide whether a candidate
 * point should emit a particle with respect to a geometry instance.
 *
 * Typical uses include:
 * - emitting particles only from a surface shell,
 * - emitting particles from the interior of a volume,
 * - switching between spawn rules without changing higher-level source logic.
 *
 * ## Design
 * The concrete spawn policies are represented as lightweight stateless operator
 * types. The public @ref SpawnOperator wrapper stores one active policy selected
 * by a runtime @ref SpawnType tag.
 *
 * This design allows:
 * - host/device portability,
 * - value-type storage in buffers,
 * - runtime dispatch without virtual inheritance.
 *
 * ## Tolerance
 * Both spawn policies accept an optional geometric tolerance parameter. This can
 * be used to:
 * - widen surface classification bands,
 * - soften strict boundary checks,
 * - improve numerical robustness near geometry boundaries.
 *
 * ## Host/device usage
 * All query-facing entry points are marked `ATLAS_ALL_DEVICE`, allowing spawn
 * classification to be used consistently in:
 * - host-side code,
 * - device kernels,
 * - backend-portable source logic.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and tolerances.
 */

#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Selects the geometric spawning rule used by a source.
 *
 * @details
 * This enum identifies which concrete spawn policy is active in a
 * @ref SpawnOperator.
 */
enum class SpawnType : int {
    /**
     * @brief Spawn relative to a geometry surface.
     *
     * @details
     * Candidate points are classified according to whether they lie on or near
     * the geometry surface, subject to the supplied tolerance.
     */
    Surface,

    /**
     * @brief Spawn relative to a geometry volume.
     *
     * @details
     * Candidate points are classified according to whether they lie inside the
     * geometry volume, subject to the supplied tolerance.
     */
    Volume
};

/**
 * @brief Surface-based spawning policy.
 *
 * @details
 * @ref SurfaceSpawnOperator classifies a candidate point with respect to the
 * surface of a geometry object.
 *
 * This policy is typically used when particles should be emitted:
 * - directly from a boundary,
 * - from a thin shell around a surface,
 * - from a geometric interface rather than the interior.
 *
 * The exact classification rule is implementation-defined in
 * `spawn_operator.hpp`, but it typically relies on:
 * - signed-distance evaluation,
 * - surface classification,
 * - a tolerance band around the surface.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and tolerances.
 */
template <typename T>
struct SurfaceSpawnOperator final {
    /**
     * @brief Return whether a candidate point should spawn according to the surface rule.
     *
     * @details
     * Evaluates the candidate point against the supplied geometry query object
     * and decides whether it belongs to the active spawning region associated
     * with the geometry surface.
     *
     * @param query Geometry operator used for surface classification.
     * @param particle Candidate particle/sample position.
     * @param tolerance Optional geometric tolerance applied to the surface test.
     * @return `true` if the candidate point should spawn; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

/**
 * @brief Volume-based spawning policy.
 *
 * @details
 * @ref VolumeSpawnOperator classifies a candidate point with respect to the
 * interior of a geometry object.
 *
 * This policy is typically used when particles should be emitted:
 * - throughout the interior of a region,
 * - from filled volumes,
 * - from sampling grids clipped by a solid domain.
 *
 * The exact classification rule is implementation-defined in
 * `spawn_operator.hpp`, but it typically relies on:
 * - inside/outside classification,
 * - signed-distance evaluation,
 * - an optional tolerance around the interior boundary.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for coordinates and tolerances.
 */
template <typename T>
struct VolumeSpawnOperator final {
    /**
     * @brief Return whether a candidate point should spawn according to the volume rule.
     *
     * @details
     * Evaluates the candidate point against the supplied geometry query object
     * and decides whether it belongs to the active spawning region associated
     * with the geometry interior.
     *
     * @param query Geometry operator used for volume classification.
     * @param particle Candidate particle/sample position.
     * @param tolerance Optional geometric tolerance applied to the volume test.
     * @return `true` if the candidate point should spawn; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

/**
 * @brief Tagged-union wrapper over all supported spawn policies.
 *
 * @details
 * @ref SpawnOperator stores one active concrete spawn policy together with a
 * runtime @ref SpawnType tag identifying which policy is currently active.
 *
 * The wrapper provides:
 * - default construction,
 * - tag-based construction,
 * - copy construction and copy assignment,
 * - explicit construction from concrete policy objects,
 * - a single dispatching @ref spawn entry point.
 *
 * ## Active state
 * The active union member is determined by @ref type:
 * - `SpawnType::Surface` activates @ref surface,
 * - `SpawnType::Volume` activates @ref volume.
 *
 * ## Intended use
 * This wrapper is designed for:
 * - storing spawn rules in buffers,
 * - passing spawn policies into host/device code,
 * - runtime selection of spawn behavior without polymorphic allocation.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the spawn rules.
 */
template <typename T>
struct SpawnOperator final {
    /**
     * @brief Active spawn-policy tag.
     *
     * @details
     * Determines which union member is currently active.
     */
    SpawnType type = SpawnType::Surface;

    /**
     * @brief Union storing the active concrete spawn policy.
     *
     * @details
     * Only the member corresponding to @ref type is considered active.
     */
    union {
        /**
         * @brief Surface-based spawn policy.
         */
        SurfaceSpawnOperator<T> surface;

        /**
         * @brief Volume-based spawn policy.
         */
        VolumeSpawnOperator<T> volume;
    };

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a spawn operator with the default active type and corresponding
     * default-initialized union member.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator() noexcept;

    /**
     * @brief Construct a spawn operator from a runtime spawn type.
     *
     * @details
     * Activates the union member corresponding to @p type.
     *
     * @param type Active spawn-policy tag.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SpawnOperator(SpawnType type) noexcept;

    /**
     * @brief Copy constructor.
     *
     * @details
     * Copies the active tag and reconstructs the corresponding active union member.
     *
     * @param other Source spawn operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator(const SpawnOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Replaces the current active state with the active state of @p other.
     *
     * @param other Source spawn operator.
     * @return `*this`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SpawnOperator&
    operator=(const SpawnOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * @details
     * Destroys the currently active union member.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SpawnOperator() noexcept;

    /**
     * @brief Construct a spawn operator from a concrete surface policy.
     *
     * @param op Source surface spawn operator.
     */
    ATLAS_HOST
    SpawnOperator(const SurfaceSpawnOperator<T>& op);

    /**
     * @brief Construct a spawn operator from a concrete volume policy.
     *
     * @param op Source volume spawn operator.
     */
    ATLAS_HOST
    SpawnOperator(const VolumeSpawnOperator<T>& op);

    /**
     * @brief Dispatch spawning classification to the active concrete policy.
     *
     * @details
     * Forwards the query to the currently active spawn operator selected by @ref type.
     *
     * @param query Geometry operator used for classification.
     * @param particle Candidate particle/sample position.
     * @param tolerance Optional geometric tolerance applied to the active test.
     * @return `true` if the candidate point satisfies the active spawn rule; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
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
     * @brief Copy the active state from another spawn operator.
     *
     * @details
     * Uses the source operator's @ref type to reconstruct the corresponding
     * active union member in this object.
     *
     * @param other Source spawn operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SpawnOperator& other) noexcept;
};

} // namespace atlas::system

#include <atlas/source/spawn_operator.hpp>