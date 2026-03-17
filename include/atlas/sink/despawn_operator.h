#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/geometry/query_operator.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Despawn classification mode used by @ref DespawnOperator.
 *
 * @tparam T Floating-point scalar type.
 */
enum class DespawnType : int {
    Surface,
    Volume
};

/**
 * @brief Collects indices of particles lying on a query operator surface.
 *
 * @details
 * Each particle position is tested with `query.is_on_surface(...)`. Matching particle
 * indices are compacted into `despawn_indices`.
 *
 * @note
 * The output buffer is always a `DeviceBuffer<int>` so the result remains directly usable
 * by CUDA/TBB backends without changing the public API shape.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceDespawnOperator final {
    ATLAS_ALL_DEVICE void
    despawn(DeviceBuffer<int>& despawn_indices,
            const DeviceBuffer<Vector3<T>>& particles,
            const atlas::geometry::QueryOperator<T>& query,
            T tolerance = T(0)) const;
};

/**
 * @brief Collects indices of particles lying inside a query operator volume.
 *
 * @details
 * Each particle position is tested with `query.is_inside(...)`. Matching particle
 * indices are compacted into `despawn_indices`.
 *
 * @note
 * The output buffer is always a `DeviceBuffer<int>` so the result remains directly usable
 * by CUDA/TBB backends without changing the public API shape.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct VolumeDespawnOperator final {
    ATLAS_ALL_DEVICE void
    despawn(DeviceBuffer<int>& despawn_indices,
            const DeviceBuffer<Vector3<T>>& particles,
            const atlas::geometry::QueryOperator<T>& query,
            T tolerance = T(0)) const;
};

/**
 * @brief Runtime-dispatched despawn operator for surface or volume classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct DespawnOperator final {
    DespawnType type = DespawnType::Surface;

    ATLAS_ALL_DEVICE
    DespawnOperator(DespawnType type)
        : type(type) { }

    ATLAS_ALL_DEVICE void
    despawn(DeviceBuffer<int>& despawn_indices,
            const DeviceBuffer<Vector3<T>>& particles,
            const atlas::geometry::QueryOperator<T>& query,
            T tolerance = T(0)) const;
};

} // namespace atlas::system

#include <atlas/sink/despawn_operator.hpp>
