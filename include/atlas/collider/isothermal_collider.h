#pragma once

#include <atlas/collider/diffuse_sampling.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>
#include <atlas/spatial/ray.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

class IsothermalCollider final {
public:
    class Builder;

public:
    IsothermalCollider() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    IsothermalCollider(Unit unit,
                       const float momentum_accommodation_coefficient,
                       const float restitution,
                       const DiffuseSampling diffuse_sampling) noexcept
        : _unit(std::move(unit))
        , _momentum_accommodation_coefficient(momentum_accommodation_coefficient)
        , _restitution(restitution)
        , _diffuse_sampling(diffuse_sampling)
        , _bound(_unit.world_bound()) {
    }

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Unit&
    unit() const noexcept {
        return _unit;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    momentum_accommodation_coefficient() const noexcept {
        return _momentum_accommodation_coefficient;
    }

    // The unit only moves in advance(), so its world bound is cached rather
    // than rebuilt (8 corners through the sync transform) on every query.
    // Invalid for an unbounded geometry such as a plane.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB&
    bound() const noexcept {
        return _bound;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(const float dt) noexcept {
        _unit.update(dt);
        _bound = _unit.world_bound();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Float3& position, const Float3& velocity, const float dt) const noexcept {
        const float speed        = velocity.length();
        const float sweep_length = speed * dt;

        if (sweep_length <= atlas::eps || speed <= atlas::eps) {
            return HitSurface {};
        }

        const atlas::Ray world_ray(position, velocity * dt);
        const HitSurface hit = _unit.trace(world_ray);

        if (!hit.is_intersecting || hit.distance > sweep_length) {
            return HitSurface {};
        }

        return hit;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    collide(const HitSurface& hit, Float3& position, Float3& velocity, const float) const noexcept {
        if (!hit.is_intersecting) {
            return;
        }

        const Float3 wall_velocity     = _unit.surface_velocity(hit.point);
        const Float3 relative_incident = velocity - wall_velocity;

        position = hit.point + hit.normal * atlas::tol;
        velocity = reflect(relative_incident, hit.normal) + wall_velocity;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    reflect(const Float3& incident, const Float3& normal) const noexcept {
        const float incident_speed = incident.length();

        if (incident_speed <= atlas::tol) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        const Float3 specular_unit = atlas::reflected(incident, normal).normalized();

        if (_momentum_accommodation_coefficient <= 0.0f) {
            return specular_unit * (incident_speed * _restitution);
        }

        const float u1 = atlas::sample_hashed_unit_interval(
            incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U1);

        const float u2 = atlas::sample_hashed_unit_interval(
            normal + incident,
            atlas::RANDOM_HASH_SALT_DIFFUSE_U2);

        Float3 diffuse_dir {};

        if (_diffuse_sampling == DiffuseSampling::cosine_weighted) {
            diffuse_dir = atlas::sample_cosine_hemisphere(normal, u1, u2);
        } else {
            diffuse_dir = atlas::sample_uniform_hemisphere(normal, u1, u2);
        }

        const float mix = atlas::sample_hashed_unit_interval(
            incident + normal * atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX,
            atlas::RANDOM_HASH_SALT_MIX);

        const Float3 out_unit = (mix < _momentum_accommodation_coefficient)
            ? diffuse_dir.normalized()
            : specular_unit;

        return out_unit * (incident_speed * _restitution);
    }

private:
    Unit _unit;

    float _momentum_accommodation_coefficient { 1.0f };

    float _restitution { 1.0f };

    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };

    AABB _bound {};
};

class IsothermalCollider::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_unit(Unit unit);

    ATLAS_HOST Builder&
    with_momentum_accommodation_coefficient(float momentum_accommodation_coefficient) noexcept;

    ATLAS_HOST Builder&
    with_restitution(float restitution) noexcept;

    ATLAS_HOST Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_NODISCARD ATLAS_HOST IsothermalCollider
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<IsothermalCollider>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Unit> _unit;

    float _momentum_accommodation_coefficient { 1.0f };

    float _restitution { 1.0f };

    DiffuseSampling _diffuse_sampling { DiffuseSampling::uniform };
};

using IsothermalColliderHostPtr = atlas::host_shared_ptr<IsothermalCollider>;

using IsothermalColliderDevicePtr = atlas::device_shared_ptr<IsothermalCollider>;

}
