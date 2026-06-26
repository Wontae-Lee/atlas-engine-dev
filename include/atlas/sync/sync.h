#pragma once

#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sync/sync_operator.h>

namespace atlas {

template <typename T>
class Sync final {
public:
    class Builder;

    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE constexpr Sync(const Vector3<T>& translation_,
                                                 const Quaternion<T>& orientation_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Sync(const atlas::SyncOperator<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~Sync() = default;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_world(const Vector3<T>& local_point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_to_local(const Vector3<T>& world_point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_world(const Vector3<T>& local_dir) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sync_dir_to_local(const Vector3<T>& world_dir) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_world(const Ray<T>& local_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray<T>
    sync_to_local(const Ray<T>& world_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray<T>& local_ray,
                  Ray<T>& world_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray<T>& world_ray,
                  Ray<T>& local_ray) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Vector3<T>& translation_) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion<T>& orientation_) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Vector3<T>& translation_,
             const Quaternion<T>& orientation_) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::SyncOperator<T>&
    sync() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE atlas::SyncOperator<T>
    make_sync_operator() const noexcept;

private:
    friend class Builder;

private:
    atlas::SyncOperator<T> sync_operator;
};

template <typename T>
class Sync<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rigid_pose(const Vector3<T>& translation_,
                    const Quaternion<T>& orientation_) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync_operator(const atlas::SyncOperator<T>& op) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Sync<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Sync<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    Vector3<T> _translation { T(0), T(0), T(0) };

    Quaternion<T> _orientation { T(1), T(0), T(0), T(0) };

    bool _has_sync_operator = false;

    atlas::SyncOperator<T> _operator {};
};

}

namespace atlas {

template <typename T>
using SyncHostPtr = atlas::host_shared_ptr<atlas::Sync<T>>;

template <typename T>
using SyncDevicePtr = atlas::device_shared_ptr<atlas::Sync<T>>;

}

#include <atlas/sync/sync.hpp>