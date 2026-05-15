#pragma once

/**
 * @file surface_despawn_operator.h
 * @brief Declares the surface-based particle despawn policy.
 */

#include <atlas/geometry/geometry_operator.h>

namespace atlas::fluid {

/**
 * @brief Stateless despawn policy that removes particles on a geometry surface.
 *
 * This policy evaluates whether a candidate particle position lies on the
 * surface of the queried geometry within the specified tolerance.
 *
 * @tparam T Floating-point scalar type used for geometry queries.
 */
template <typename T>
struct SurfaceDespawnOperator final {

    /**
     * @brief Tests whether a particle should be removed based on surface membership.
     *
     * The particle is considered removable when it lies on the queried geometry
     * surface within the provided tolerance.
     *
     * @param query Geometry query operator used to test spatial membership.
     * @param particle Candidate particle position in query space.
     * @param tolerance Surface-membership tolerance.
     * @return True if the particle lies on the surface within tolerance.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::geometry::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

} // namespace atlas::fluid

#include <atlas/sink/surface_despawn_operator.hpp>
