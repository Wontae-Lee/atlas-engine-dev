#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/geometry/query_operator.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Spawn classification mode used by @ref SpawnOperator.
 *
 * @tparam T Floating-point scalar type.
 */
enum class SpawnType : int {
    Surface,
    Volume
};

/**
 * @brief Builds particles on a regular grid around a query operator surface.
 *
 * @details
 * The operator samples the finite AABB returned by `query.bound()` using `spacing` as the
 * Cartesian grid interval. Each grid point is tested with `query.is_on_surface(...)`.
 * Points that pass the predicate are written to `particles`.
 *
 * The optional `tolerance` defaults to `spacing * 0.5`, which forms a surface band wide enough
 * to catch the surface on a regular grid.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceSpawnOperator final {
    ATLAS_ALL_DEVICE void
    spawn(DeviceBuffer<Vector3<T>>& particles,
          const atlas::geometry::QueryOperator<T>& query,
          T spacing,
          T tolerance = T(-1)) const;
};

/**
 * @brief Builds particles on a regular grid inside a query operator volume.
 *
 * @details
 * The operator samples the finite AABB returned by `query.bound()` using `spacing` as the
 * Cartesian grid interval. Each grid point is tested with `query.is_inside(...)`.
 * Points that pass the predicate are written to `particles`.
 *
 * The optional `tolerance` defaults to zero.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct VolumeSpawnOperator final {
    ATLAS_ALL_DEVICE void
    spawn(DeviceBuffer<Vector3<T>>& particles,
          const atlas::geometry::QueryOperator<T>& query,
          T spacing,
          T tolerance = T(0)) const;
};

/**
 * @brief Runtime-dispatched spawn operator for surface or volume sampling.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SpawnOperator final {
    SpawnType type = SpawnType::Surface;

    ATLAS_ALL_DEVICE void
    spawn(DeviceBuffer<Vector3<T>>& particles,
          const atlas::geometry::QueryOperator<T>& query,
          T spacing,
          T tolerance = T(-1)) const;
};

} // namespace atlas::system

#include <atlas/source/spawn_operator.hpp>
