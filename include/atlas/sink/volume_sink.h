#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <utility>

namespace atlas {

class VolumeSink final {
public:
    class Builder;

public:
    VolumeSink() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    VolumeSink(Unit unit, const float tolerance) noexcept
        : _unit(std::move(unit))
        , _tolerance(tolerance) {
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
    despawn(const Float3& position, const Float3&, const float) const noexcept {
        const Float3 local = _unit.sync().sync_to_local(position);
        return _unit.geometry().is_inside(local, _tolerance);
    }

private:
    Unit _unit;

    float _tolerance { 0.0f };
};

class VolumeSink::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_unit(Unit unit);

    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    ATLAS_NODISCARD ATLAS_HOST VolumeSink
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<VolumeSink>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Unit> _unit;

    float _tolerance { 0.0f };
};

using VolumeSinkHostPtr = atlas::host_shared_ptr<VolumeSink>;

using VolumeSinkDevicePtr = atlas::device_shared_ptr<VolumeSink>;

}
