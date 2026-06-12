#pragma once

/**
 * @file volume_despawn_operator.h
 * @brief Declares the volume-based particle despawn policy.
 */

#include <atlas/geometry/geometry_operator.h>

namespace atlas {

/**
 * @brief Stateless despawn policy that removes particles inside a geometry region.
 *
 * This policy evaluates whether a candidate particle position lies inside
 * the queried geometry volume or region within the specified tolerance.
 *
 * @tparam T Floating-point scalar type used for geometry queries.
 */
template <typename T>
struct VolumeDespawnOperator final {

    /**
     * @brief Tests whether a particle should be removed based on interior membership.
     *
     * The particle is considered removable when it lies inside the queried
     * geometry region within the provided tolerance.
     *
     * @param query Geometry query operator used to test spatial membership.
     * @param particle Candidate particle position in query space.
     * @param tolerance Interior-membership tolerance.
     * @return True if the particle lies inside within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

} // namespace atlas

#include <atlas/sink/volume_despawn_operator.hpp>
