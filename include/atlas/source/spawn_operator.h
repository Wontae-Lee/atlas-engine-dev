#pragma once

/**
 * @file spawn_operator.h
 * @brief Declares spawn operator types used to test whether particle candidates are accepted for emission.
 */

#include <atlas/math/math.h>

namespace atlas::fluid {

/**
 * @brief Runtime tag identifying the spawn acceptance rule.
 */
enum class SpawnType : int {
    /**
     * @brief Accept particles located on the queried geometry surface.
     */
    Surface,

    /**
     * @brief Accept particles located inside the queried geometry volume or region.
     */
    Volume
};

/**
 * @brief Spawn policy that accepts particles on a geometry surface.
 *
 * This operator evaluates whether a candidate particle position lies on the
 * surface of the queried geometry within a specified tolerance.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceSpawnOperator final {

    /**
     * @brief Tests whether a particle position lies on the queried surface.
     *
     * @param query Geometry query operator.
     * @param particle Candidate particle position.
     * @param tolerance Surface tolerance.
     * @return True if the particle is on the surface within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

/**
 * @brief Spawn policy that accepts particles inside a geometry region.
 *
 * This operator evaluates whether a candidate particle position lies inside
 * the queried geometry within a specified tolerance.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct VolumeSpawnOperator final {

    /**
     * @brief Tests whether a particle position lies inside the queried region.
     *
     * @param query Geometry query operator.
     * @param particle Candidate particle position.
     * @param tolerance Interior tolerance.
     * @return True if the particle is inside within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

/**
 * @brief Tagged spawn operator that dispatches between surface and volume policies.
 *
 * This type stores exactly one active spawn policy at a time and dispatches
 * spawn tests according to the runtime tag stored in @ref type.
 *
 * Internally, it manages a union of:
 * - SurfaceSpawnOperator<T>
 * - VolumeSpawnOperator<T>
 *
 * Because the active union member is selected dynamically, this type manually
 * manages construction, destruction, and copying of the active member.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SpawnOperator final {
    /**
     * @brief Runtime tag indicating which spawn policy is active.
     */
    SpawnType type = SpawnType::Surface;

    /**
     * @brief Storage for the active concrete spawn policy.
     *
     * Exactly one member is active at a time, as indicated by @ref type.
     */
    union {
        /**
         * @brief Surface-based spawn policy storage.
         */
        SurfaceSpawnOperator<T> surface;

        /**
         * @brief Volume-based spawn policy storage.
         */
        VolumeSpawnOperator<T> volume;
    };

    /**
     * @brief Default constructor.
     *
     * Initializes the operator with a surface-based spawn policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator() noexcept;

    /**
     * @brief Constructs a spawn operator of the requested type.
     *
     * @param type Runtime spawn policy type to activate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SpawnOperator(SpawnType type) noexcept;

    /**
     * @brief Copy constructor.
     *
     * Copies the runtime tag and reconstructs the corresponding active policy.
     *
     * @param other Source operator to copy from.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator(const SpawnOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * Replaces the current active policy with a copy of the one stored in @p other.
     *
     * @param other Source operator to copy from.
     * @return Reference to this object.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SpawnOperator&
    operator=(const SpawnOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * Destroys the currently active concrete spawn policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SpawnOperator() noexcept;

    /**
     * @brief Constructs the operator from a surface spawn policy.
     *
     * The runtime tag is set to @ref SpawnType::Surface.
     *
     * @param op Concrete surface spawn policy.
     */
    ATLAS_HOST
    SpawnOperator(const SurfaceSpawnOperator<T>& op);

    /**
     * @brief Constructs the operator from a volume spawn policy.
     *
     * The runtime tag is set to @ref SpawnType::Volume.
     *
     * @param op Concrete volume spawn policy.
     */
    ATLAS_HOST
    SpawnOperator(const VolumeSpawnOperator<T>& op);

    /**
     * @brief Evaluates whether a candidate particle should be emitted.
     *
     * Dispatches the query to the currently active spawn policy.
     *
     * @param query Geometry query operator.
     * @param particle Candidate particle position.
     * @param tolerance Tolerance used by the active spawn policy.
     * @return True if the particle is accepted for spawning.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) const noexcept;

private:
    /**
     * @brief Destroys the currently active concrete spawn policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    /**
     * @brief Reconstructs the active policy from another operator.
     *
     * This function assumes that @ref type has already been set to the desired
     * active tag before it is called.
     *
     * @param other Source operator providing the active policy to copy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SpawnOperator& other) noexcept;
};

} // namespace atlas::fluid

#include <atlas/source/spawn_operator.hpp>