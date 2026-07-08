#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/source/source_type.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>

#include <concepts>
#include <cstddef>
#include <new>
#include <utility>

namespace atlas {

class FluidPositionState;

template <typename S>
concept ConceptSource = requires(S source, const S const_source, FluidPositionState* positions, std::size_t offset, float dt) {
    { const_source.spawn(positions, offset) } -> std::same_as<int>;
    { source.advance(dt) } -> std::same_as<void>;
};

static_assert(ConceptSource<SurfaceSource>);
static_assert(ConceptSource<VolumeSource>);

// Host-side tagged union over the concrete source leaves. Unlike Sink/Collider
// it is not a DeviceVariant: each leaf owns a DeviceBuffer<Float3> cache, so the
// union is host-only and move-only (its special members are written by hand).
class Source final {
public:
    SourceType type = SourceType::surface;

    union {

        SurfaceSource surface;

        VolumeSource volume;
    };

    ATLAS_HOST
    Source() noexcept
        : type(SourceType::surface)
        , surface() {
    }

    ATLAS_HOST explicit
    Source(SurfaceSource op) noexcept
        : type(SourceType::surface)
        , surface(std::move(op)) {
    }

    ATLAS_HOST explicit
    Source(VolumeSource op) noexcept
        : type(SourceType::volume)
        , volume(std::move(op)) {
    }

    Source(const Source&) = delete;

    Source&
    operator=(const Source&)
        = delete;

    ATLAS_HOST
    Source(Source&& other) noexcept
        : type(other.type) {
        construct_from(std::move(other));
    }

    ATLAS_HOST Source&
    operator=(Source&& other) noexcept {
        if (this != &other) {
            destroy();
            type = other.type;
            construct_from(std::move(other));
        }
        return *this;
    }

    ATLAS_HOST
    ~Source() noexcept {
        destroy();
    }

    ATLAS_HOST void
    advance(const float dt) noexcept {
        switch (type) {
            case SourceType::surface:
                surface.advance(dt);
                break;
            case SourceType::volume:
                volume.advance(dt);
                break;
        }
    }

    ATLAS_NODISCARD ATLAS_HOST int
    spawn(FluidPositionState* positions, const std::size_t offset) const {
        switch (type) {
            case SourceType::surface:
                return surface.spawn(positions, offset);
            case SourceType::volume:
                return volume.spawn(positions, offset);
        }
        return 0;
    }

private:
    ATLAS_HOST void
    construct_from(Source&& other) noexcept {
        switch (type) {
            case SourceType::surface:
                new (&surface) SurfaceSource(std::move(other.surface));
                break;
            case SourceType::volume:
                new (&volume) VolumeSource(std::move(other.volume));
                break;
        }
    }

    ATLAS_HOST void
    destroy() noexcept {
        switch (type) {
            case SourceType::surface:
                surface.~SurfaceSource();
                break;
            case SourceType::volume:
                volume.~VolumeSource();
                break;
        }
    }
};

using SourceHostPtr = atlas::host_shared_ptr<Source>;

using SourceDevicePtr = atlas::device_shared_ptr<Source>;

}
