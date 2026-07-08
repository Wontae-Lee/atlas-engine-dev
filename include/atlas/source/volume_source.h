#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace atlas {

// Caches the accepted interior sample points of its unit (in the unit's local
// frame) and emits them, transformed to world space via the unit's sync, into
// a fluid position buffer.
class VolumeSource final {
public:
    class Builder;

public:
    VolumeSource() = default;

    ATLAS_HOST
    VolumeSource(Unit unit, float tolerance, float spacing);

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_NODISCARD ATLAS_HOST const Unit&
    unit() const noexcept {
        return _unit;
    }

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    cached_count() const noexcept {
        return _cache.size();
    }

    ATLAS_HOST void
    advance(const float dt) noexcept {
        _unit.update(dt);
    }

    ATLAS_NODISCARD ATLAS_HOST int
    spawn(FluidPositionState* positions, std::size_t offset) const;

private:
    Unit _unit;

    float _tolerance { 0.0f };

    float _spacing { 0.1f };

    DeviceBuffer<Float3> _cache;
};

class VolumeSource::Builder final {
public:
    Builder() = default;

    ATLAS_HOST Builder&
    with_unit(Unit unit);

    ATLAS_HOST Builder&
    with_tolerance(float tolerance) noexcept;

    ATLAS_HOST Builder&
    with_spacing(float spacing) noexcept;

    ATLAS_NODISCARD ATLAS_HOST VolumeSource
    build();

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<VolumeSource>
    make_host_shared();

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<Unit> _unit;

    float _tolerance { 0.0f };

    float _spacing { 0.1f };
};

using VolumeSourceHostPtr = atlas::host_shared_ptr<VolumeSource>;

using VolumeSourceDevicePtr = atlas::device_shared_ptr<VolumeSource>;

}
