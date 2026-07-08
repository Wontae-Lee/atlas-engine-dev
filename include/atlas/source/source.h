#pragma once

#include <atlas/core/host_variant.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/memory.h>
#include <atlas/source/source_type.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>

#include <concepts>
#include <cstddef>
#include <utility>

namespace atlas {

template <typename S>
concept ConceptSource = requires(S source, const S const_source, FluidPositionState* positions, std::size_t offset, float dt) {
    { const_source.spawn(positions, offset) } -> std::same_as<int>;
    { source.advance(dt) } -> std::same_as<void>;
};

static_assert(ConceptSource<SurfaceSource>);
static_assert(ConceptSource<VolumeSource>);

// Host-side tagged union over the concrete source leaves. Each leaf owns a
// DeviceBuffer<Float3> cache, so this uses HostVariant (host-only, move-based)
// rather than a DeviceVariant.
class Source final {
public:
    SourceType type = SourceType::surface;

    union {

        SurfaceSource surface;

        VolumeSource volume;
    };

    ATLAS_HOST
    Source() noexcept;

    ATLAS_HOST explicit
    Source(SurfaceSource op) noexcept;

    ATLAS_HOST explicit
    Source(VolumeSource op) noexcept;

    Source(const Source&) = delete;

    Source&
    operator=(const Source&)
        = delete;

    ATLAS_HOST
    Source(Source&& other) noexcept;

    ATLAS_HOST Source&
    operator=(Source&& other) noexcept;

    ATLAS_HOST
    ~Source() noexcept;

    ATLAS_HOST void
    advance(float dt) noexcept;

    ATLAS_NODISCARD ATLAS_HOST int
    spawn(FluidPositionState* positions, std::size_t offset) const;
};

using SourceVariant = HostVariant<
    Source,
    SourceType,
    SourceType::surface,
    HostVariantCase<SourceType::surface, &Source::surface>,
    HostVariantCase<SourceType::volume, &Source::volume>>;

class SourceAdvance {
public:
    float dt;
    template <typename S>
    ATLAS_HOST void
    operator()(S& source) const noexcept { source.advance(dt); }
};

class SourceSpawn {
public:
    FluidPositionState* positions;
    std::size_t offset;
    template <typename S>
    ATLAS_HOST int
    operator()(const S& source) const { return source.spawn(positions, offset); }
};

ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source() noexcept {
    SourceVariant::construct(*this, SourceType::surface);
}

ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source(SurfaceSource op) noexcept {
    SourceVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source(VolumeSource op) noexcept {
    SourceVariant::construct_payload(*this, std::move(op));
}

ATLAS_HOST ATLAS_FORCE_INLINE
Source::Source(Source&& other) noexcept {
    SourceVariant::move_construct(*this, std::move(other));
}

ATLAS_HOST ATLAS_FORCE_INLINE Source&
Source::operator=(Source&& other) noexcept {
    SourceVariant::move_assign(*this, std::move(other));
    return *this;
}

ATLAS_HOST ATLAS_FORCE_INLINE
Source::~Source() noexcept {
    SourceVariant::destroy(*this);
}

ATLAS_HOST ATLAS_FORCE_INLINE void
Source::advance(const float dt) noexcept {
    SourceVariant::apply(*this, SourceAdvance { dt });
}

ATLAS_HOST ATLAS_FORCE_INLINE int
Source::spawn(FluidPositionState* positions, const std::size_t offset) const {
    return SourceVariant::visit(*this, SourceSpawn { positions, offset }, 0);
}

using SourceHostPtr = atlas::host_shared_ptr<Source>;

using SourceDevicePtr = atlas::device_shared_ptr<Source>;

}
