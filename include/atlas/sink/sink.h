#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/sink/sink_type.h>
#include <atlas/sink/surface_sink.h>
#include <atlas/sink/tracing_sink.h>
#include <atlas/sink/volume_sink.h>

#include <concepts>
#include <type_traits>

namespace atlas {

template <typename S>
concept ConceptSink = requires(S sink, const Float3 vec, float dt) {
    { sink.despawn(vec, vec, dt) } -> std::same_as<bool>;
    { sink.advance(dt) } -> std::same_as<void>;
};

static_assert(ConceptSink<SurfaceSink>);
static_assert(ConceptSink<VolumeSink>);
static_assert(ConceptSink<TracingSink>);

class Sink final {
public:

    SinkType type = SinkType::surface;

    union {

        SurfaceSink surface;

        VolumeSink volume;

        TracingSink tracing;
    };

    ATLAS_ALL_DEVICE
    Sink() noexcept;

    ATLAS_ALL_DEVICE
    Sink(const Sink& other) noexcept = default;

    ATLAS_ALL_DEVICE Sink&
    operator=(const Sink& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~Sink() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Sink>, int> = 0>
    ATLAS_ALL_DEVICE explicit Sink(const Payload& op) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(float dt) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const Float3& position, const Float3& velocity, float dt) const noexcept;
};

using SinkVariant = DeviceVariant<
    Sink,
    SinkType,
    SinkType::surface,
    DeviceVariantCase<SinkType::surface, &Sink::surface>,
    DeviceVariantCase<SinkType::volume, &Sink::volume>,
    DeviceVariantCase<SinkType::tracing, &Sink::tracing>>;

class SinkAdvance {
public:
    float dt;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(S& sink) const noexcept { sink.advance(dt); }
};

class SinkDespawn {
public:
    const Float3& position;
    const Float3& velocity;
    float dt;
    template <typename S>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(const S& sink) const noexcept { return sink.despawn(position, velocity, dt); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Sink::Sink() noexcept {
    SinkVariant::construct(*this, SinkType::surface);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Sink>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Sink::Sink(const Payload& op) noexcept {
    SinkVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
Sink::advance(const float dt) noexcept {
    SinkVariant::apply(*this, SinkAdvance { dt });
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Sink::despawn(const Float3& position, const Float3& velocity, const float dt) const noexcept {
    return SinkVariant::visit(
        *this,
        SinkDespawn { position, velocity, dt },
        false);
}

using SinkHostPtr = atlas::host_shared_ptr<Sink>;

using SinkDevicePtr = atlas::device_shared_ptr<Sink>;

}
