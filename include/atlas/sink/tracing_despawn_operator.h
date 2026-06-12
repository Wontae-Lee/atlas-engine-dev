#pragma once

/**
 * @file tracing_despawn_operator.h
 * @brief Declares the velocity-tracing particle despawn policy.
 */

#include <atlas/geometry/geometry_operator.h>
#include <atlas/spatial/ray.h>

namespace atlas {

/**
 * @brief Stateless despawn policy that traces a velocity vector through geometry.
 *
 * This policy traces a particle from its current position along its velocity
 * over a time interval. A particle is removable when that segment intersects
 * the queried geometry.
 *
 * @tparam T Floating-point scalar type used for geometry queries.
 */
template <typename T>
struct TracingDespawnOperator final {

    /**
     * @brief Tests whether the velocity trace intersects the queried geometry.
     *
     * @param query Geometry query operator used to trace the velocity.
     * @param position Candidate particle position in query-local space.
     * @param velocity Candidate particle velocity in query-local space.
     * @param time Time interval used as the maximum ray parameter.
     * @return True if the velocity trace intersects the geometry within @p time.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& position,
            const Vector3<T>& velocity,
            T time = T(0)) noexcept;
};

} // namespace atlas

#include <atlas/sink/tracing_despawn_operator.hpp>
