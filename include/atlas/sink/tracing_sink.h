#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/ray.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

class TracingSink final {
public:
    class Builder;

public:
    TracingSink() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit TracingSink(Unit unit) noexcept
        : _unit(std::move(unit)) {
    }

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Unit&
    unit() const noexcept {
        return _unit;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(const float dt) noexcept {
        _unit.update(dt);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const Float3& position, const Float3& velocity, const float dt) const noexcept {
        const float speed = velocity.length();

        if (!(dt > 0.0f) || !(speed > 0.0f)) {
            return false;
        }

        const HitSurface hit = _unit.trace(atlas::Ray(position, velocity));

        return hit.is_intersecting && hit.distance >= 0.0f && hit.distance <= speed * dt;
    }

private:
    Unit _unit;
};

class TracingSink::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_unit(Unit unit);

    ATLAS_NODISCARD ATLAS_HOST TracingSink
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<TracingSink>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Unit> _unit;
};

using TracingSinkHostPtr = atlas::host_shared_ptr<TracingSink>;

using TracingSinkDevicePtr = atlas::device_shared_ptr<TracingSink>;

}