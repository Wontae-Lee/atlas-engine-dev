#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/ray.h>

namespace atlas {

class Sync final {
public:
    class Builder;

public:
    Vector3 translation;
    Quaternion orientation;
    Matrix3x3 orientation_matrix;
    Matrix3x3 inverse_orientation_matrix;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Sync() noexcept
        : translation(0.0f, 0.0f, 0.0f)
        , orientation()
        , orientation_matrix(atlas::identity3x3())
        , inverse_orientation_matrix(atlas::identity3x3()) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Sync(const Vector3& translation_, const Quaternion& orientation_) noexcept
        : translation(translation_)
        , orientation(orientation_)
        , orientation_matrix()
        , inverse_orientation_matrix() {
        rebuild_matrices();
    }

    Sync(const Sync&) noexcept            = default;
    Sync(Sync&&) noexcept                 = default;
    Sync& operator=(const Sync&) noexcept = default;
    Sync& operator=(Sync&&) noexcept      = default;

    ~Sync() noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rebuild_matrices() noexcept {
        orientation_matrix         = orientation.to_matrix3x3();
        inverse_orientation_matrix = transpose(orientation_matrix);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Vector3& local_point, Vector3& world_point) const noexcept {
        atlas::rotate_translate(orientation_matrix, local_point, translation, world_point);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Vector3& world_point, Vector3& local_point) const noexcept {
        atlas::rotate_subtract(inverse_orientation_matrix, world_point, translation, local_point);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_world(const Vector3& local_dir, Vector3& world_dir) const noexcept {
        atlas::rotate(orientation_matrix, local_dir, world_dir);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_dir_to_local(const Vector3& world_dir, Vector3& local_dir) const noexcept {
        atlas::rotate(inverse_orientation_matrix, world_dir, local_dir);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_world(const Ray& local_ray, Ray& world_ray) const noexcept {
        atlas::rotate_translate(orientation_matrix, local_ray.origin, translation, world_ray.origin);
        atlas::rotate(orientation_matrix, local_ray.direction, world_ray.direction);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sync_to_local(const Ray& world_ray, Ray& local_ray) const noexcept {
        atlas::rotate_subtract(inverse_orientation_matrix, world_ray.origin, translation, local_ray.origin);
        atlas::rotate(inverse_orientation_matrix, world_ray.direction, local_ray.direction);
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    sync_to_world(const Vector3& local_point) const noexcept {
        Vector3 out;
        sync_to_world(local_point, out);
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    sync_to_local(const Vector3& world_point) const noexcept {
        Vector3 out;
        sync_to_local(world_point, out);
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    sync_dir_to_world(const Vector3& local_dir) const noexcept {
        Vector3 out;
        sync_dir_to_world(local_dir, out);
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3
    sync_dir_to_local(const Vector3& world_dir) const noexcept {
        Vector3 out;
        sync_dir_to_local(world_dir, out);
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray
    sync_to_world(const Ray& local_ray) const noexcept {
        Ray out;
        sync_to_world(local_ray, out);
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Ray
    sync_to_local(const Ray& world_ray) const noexcept {
        Ray out;
        sync_to_local(world_ray, out);
        return out;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_translation(const Vector3& translation_) noexcept {
        translation = translation_;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_orientation(const Quaternion& orientation_) noexcept {
        orientation = orientation_;
        rebuild_matrices();
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_pose(const Vector3& translation_, const Quaternion& orientation_) noexcept {
        translation = translation_;
        orientation = orientation_;
        rebuild_matrices();
    }
};

class Sync::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_rigid_pose(const Vector3& translation_, const Quaternion& orientation_) noexcept;

    ATLAS_HOST ATLAS_NODISCARD Sync
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<Sync>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    Vector3 _translation = Vector3(0.0f, 0.0f, 0.0f);

    Quaternion _orientation = Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
};

using SyncHostPtr = atlas::host_shared_ptr<Sync>;

using SyncDevicePtr = atlas::device_shared_ptr<Sync>;

}
