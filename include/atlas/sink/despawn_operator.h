#pragma once

/**
 * @file despawn_operator.h
 * @brief Declares despawn operator types used to decide whether particles should be removed.
 */

#include <atlas/math/math.h>

namespace atlas::fluid {

/**
 * @brief Runtime tag identifying the active despawn rule.
 */
enum class DespawnType : int {

    /**
     * @brief Remove particles located on the queried geometry surface.
     */
    Surface,

    /**
     * @brief Remove particles located inside the queried geometry volume or region.
     */
    Volume
};

/**
 * @brief Despawn policy that removes particles on a geometry surface.
 *
 * This operator evaluates whether a candidate particle position lies on the
 * surface of the queried geometry within a specified tolerance.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceDespawnOperator final {

    /**
     * @brief Tests whether a particle position should be removed based on surface membership.
     *
     * @param query Geometry query operator.
     * @param particle Candidate particle position.
     * @param tolerance Surface tolerance.
     * @return True if the particle is on the surface within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

/**
 * @brief Despawn policy that removes particles inside a geometry region.
 *
 * This operator evaluates whether a candidate particle position lies inside
 * the queried geometry within a specified tolerance.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct VolumeDespawnOperator final {

    /**
     * @brief Tests whether a particle position should be removed based on interior membership.
     *
     * @param query Geometry query operator.
     * @param particle Candidate particle position.
     * @param tolerance Interior tolerance.
     * @return True if the particle is inside within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

/**
 * @brief Tagged despawn operator that dispatches between surface and volume policies.
 *
 * This type stores exactly one active despawn policy at a time and dispatches
 * despawn tests according to the runtime tag stored in @ref type.
 *
 * Internally, it manages a union of:
 * - SurfaceDespawnOperator<T>
 * - VolumeDespawnOperator<T>
 *
 * Because the active union member is selected dynamically, this type manually
 * manages construction, destruction, and copying of the active member.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct DespawnOperator final {

    /**
     * @brief Runtime tag indicating which despawn policy is active.
     */
    DespawnType type = DespawnType::Surface;

    /**
     * @brief Storage for the active concrete despawn policy.
     *
     * Exactly one member is active at a time, as indicated by @ref type.
     */
    union {

        /**
         * @brief Surface-based despawn policy storage.
         */
        SurfaceDespawnOperator<T> surface;

        /**
         * @brief Volume-based despawn policy storage.
         */
        VolumeDespawnOperator<T> volume;
    };

    /**
     * @brief Default constructor.
     *
     * Initializes the operator with a surface-based despawn policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator() noexcept;

    /**
     * @brief Constructs a despawn operator of the requested type.
     *
     * @param type Runtime despawn policy type to activate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DespawnOperator(DespawnType type) noexcept;

    /**
     * @brief Copy constructor.
     *
     * Copies the runtime tag and reconstructs the corresponding active policy.
     *
     * @param other Source operator to copy from.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator(const DespawnOperator& other) noexcept;

    /**
     * @brief Copy assignment operator.
     *
     * Replaces the current active policy with a copy of the one stored in @p other.
     *
     * @param other Source operator to copy from.
     * @return Reference to this object.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DespawnOperator&
    operator=(const DespawnOperator& other) noexcept;

    /**
     * @brief Destructor.
     *
     * Destroys the currently active concrete despawn policy.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DespawnOperator() noexcept;

    /**
     * @brief Constructs the operator from a surface despawn policy.
     *
     * The runtime tag is set to @ref DespawnType::Surface.
     *
     * @param op Concrete surface despawn policy.
     */
    ATLAS_HOST
    DespawnOperator(const SurfaceDespawnOperator<T>& op);

    /**
     * @brief Constructs the operator from a volume despawn policy.
     *
     * The runtime tag is set to @ref DespawnType::Volume.
     *
     * @param op Concrete volume despawn policy.
     */
    ATLAS_HOST
    DespawnOperator(const VolumeDespawnOperator<T>& op);

    /**
     * @brief Evaluates whether a candidate particle should be removed.
     *
     * Dispatches the query to the currently active despawn policy.
     *
     * @param query Geometry query operator.
     * @param particle Candidate particle position.
     * @param tolerance Tolerance used by the active despawn policy.
     * @return True if the particle should be removed.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) const noexcept;

private:
    /**
     * @brief Destroys the currently active concrete despawn policy.
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
    copy_from(const DespawnOperator& other) noexcept;
};

} // namespace atlas::fluid

#include <atlas/sink/despawn_operator.hpp>