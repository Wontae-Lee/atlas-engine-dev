#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>
#include <atlas/sync/sync.h>

#include <optional>
#include <utility>

namespace atlas {

class Unit final {
public:
    class Builder;

public:
    Unit() = default;

    ~Unit() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(Geometry geometry,
         Sync sync) noexcept
        : _geometry(std::move(geometry))
        , _sync(std::move(sync)) {
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(Geometry geometry,
         Sync sync,
         std::optional<Float3> velocity,
         std::optional<Float3> acceleration,
         std::optional<Float3> angular_velocity,
         std::optional<Float3> angular_acceleration) noexcept
        : _geometry(std::move(geometry))
        , _sync(std::move(sync)) {
        Unit::canonicalize_kinematics(
            velocity,
            acceleration,
            angular_velocity,
            angular_acceleration);

        _velocity             = std::move(velocity);
        _acceleration         = std::move(acceleration);
        _angular_velocity     = std::move(angular_velocity);
        _angular_acceleration = std::move(angular_acceleration);
    }

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_geometry(Geometry geometry) noexcept {
        _geometry = std::move(geometry);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_sync(Sync sync) noexcept {
        _sync = std::move(sync);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    update(const float dt) noexcept {
        if (!(dt > 0.0f)) return;

        if (_velocity.has_value()) {

            if (_acceleration.has_value()) {
                *_velocity += (*_acceleration) * dt;
            }

            move((*_velocity) * dt);
        }

        if (_angular_velocity.has_value()) {

            if (_angular_acceleration.has_value()) {
                *_angular_velocity += (*_angular_acceleration) * dt;
            }

            const float omega = _angular_velocity->length();

            if (omega > 0.0f) {
                rotate(*_angular_velocity, omega * dt);
            }
        }
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    move(const Float3& delta) noexcept {
        _sync.translation += delta;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Float3& axis, const float angle_rad) noexcept {
        const float axis_len2 = axis.length_squared();

        if (axis_len2 <= 0.0f) return;

        const Float3 normalized_axis = atlas::normalized_or(
            axis,
            Float3(0.0f, 0.0f, 0.0f));

        const Quaternion rotation = Quaternion::from_axis_angle(normalized_axis, angle_rad);

        _sync.orientation = (rotation * _sync.orientation).normalized();

        _sync.rebuild_matrices();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Geometry&
    geometry() const noexcept {
        return _geometry;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Sync&
    sync() const noexcept {
        return _sync;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& world_ray) const noexcept {
        const Ray  local_ray = _sync.sync_to_local(world_ray);
        HitSurface hit       = _geometry.trace(local_ray);

        if (hit.is_intersecting) {
            hit.point  = _sync.sync_to_world(hit.point);
            hit.normal = _sync.sync_dir_to_world(hit.normal);
        }

        return hit;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    surface_velocity(const Float3& surface_point) const noexcept {
        Float3 velocity(0.0f, 0.0f, 0.0f);

        if (_velocity.has_value()) {
            velocity += *_velocity;
        }

        if (_angular_velocity.has_value()) {
            const Float3 radius = surface_point - _sync.translation;
            velocity += atlas::cross(*_angular_velocity, radius);
        }

        return velocity;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    world_bound() const noexcept {
        const AABB local_bound = _geometry.bound();

        AABB transformed {};

        if (!local_bound.is_valid()) {
            return transformed;
        }

        for (int corner = 0; corner < 8; ++corner) {
            transformed.merge(_sync.sync_to_world(local_bound.corner(static_cast<std::size_t>(corner))));
        }

        return transformed;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    velocity() const noexcept {
        return _velocity;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    acceleration() const noexcept {
        return _acceleration;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    angular_velocity() const noexcept {
        return _angular_velocity;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const std::optional<Float3>&
    angular_acceleration() const noexcept {
        return _angular_acceleration;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    dynamic() const noexcept {
        return _velocity.has_value() || _angular_velocity.has_value();
    }

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    canonicalize_kinematics(std::optional<Float3>& velocity,
                            std::optional<Float3>& acceleration,
                            std::optional<Float3>& angular_velocity,
                            std::optional<Float3>& angular_acceleration) noexcept {
        if (acceleration.has_value() && !velocity.has_value()) {
            velocity = Float3(0.0f, 0.0f, 0.0f);
        }

        if (velocity.has_value() && !acceleration.has_value()) {
            acceleration = Float3(0.0f, 0.0f, 0.0f);
        }

        if (angular_acceleration.has_value() && !angular_velocity.has_value()) {
            angular_velocity = Float3(0.0f, 0.0f, 0.0f);
        }

        if (angular_velocity.has_value() && !angular_acceleration.has_value()) {
            angular_acceleration = Float3(0.0f, 0.0f, 0.0f);
        }
    }

    friend class Builder;

private:
    Geometry _geometry;

    Sync _sync;

    std::optional<Float3> _velocity;

    std::optional<Float3> _acceleration;

    std::optional<Float3> _angular_velocity;

    std::optional<Float3> _angular_acceleration;
};

class Unit::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_geometry(const Geometry& geometry);

    ATLAS_HOST Builder&
    with_sync(const SyncHostPtr& sync);

    ATLAS_HOST Builder&
    with_velocity(const Float3& v) noexcept;

    ATLAS_HOST Builder&
    with_acceleration(const Float3& a) noexcept;

    ATLAS_HOST Builder&
    with_angular_velocity(const Float3& w) noexcept;

    ATLAS_HOST Builder&
    with_angular_acceleration(const Float3& alpha) noexcept;

    ATLAS_NODISCARD ATLAS_HOST Unit
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Unit>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Geometry> _geometry;

    std::optional<Sync> _sync;

    std::optional<Float3> _velocity;

    std::optional<Float3> _acceleration;

    std::optional<Float3> _angular_velocity;

    std::optional<Float3> _angular_acceleration;
};

}
