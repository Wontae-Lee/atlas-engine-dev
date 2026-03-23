#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

namespace atlas::system {

template <typename T>
struct SyncOperator final {

    Vector3<T> translation {};

    Quaternion<T> orientation {};

    Matrix<T, 3, 3> orientation_matrix {};

    Matrix<T, 3, 3> inverse_orientation_matrix {};

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr SyncOperator() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SyncOperator(const Vector3<T>& translation_,
                 const Quaternion<T>& orientation_) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Vector3<T>& local_point,
                  Vector3<T>& world_point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Vector3<T>& world_point,
                  Vector3<T>& local_point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Vector3<T>& local_dir,
                      Vector3<T>& world_dir) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Vector3<T>& world_dir,
                      Vector3<T>& local_dir) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const atlas::spatial::Ray<T>& local_ray,
                  atlas::spatial::Ray<T>& world_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const atlas::spatial::Ray<T>& world_ray,
                  atlas::spatial::Ray<T>& local_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_world(const atlas::spatial::Ray<T>& local_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::spatial::Ray<T>
    sync_to_local(const atlas::spatial::Ray<T>& world_ray) const noexcept;
};

}

namespace atlas {

template <typename T>
using SyncOperator = system::SyncOperator<T>;

}

#include <atlas/sync/sync_operator.hpp>