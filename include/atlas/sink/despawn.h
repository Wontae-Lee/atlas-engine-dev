#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/sink/surface_despawn.h>
#include <atlas/sink/tracing_despawn.h>
#include <atlas/sink/volume_despawn.h>

#include <type_traits>

namespace atlas {

enum class DespawnType : int {

    surface,

    volume,

    tracing
};

using DespawnTypeSwitch = DeviceTypeSwitch<
    DespawnType,
    DespawnType::surface,
    DeviceTypeCase<DespawnType, DespawnType::surface, SurfaceDespawn>,
    DeviceTypeCase<DespawnType, DespawnType::volume, VolumeDespawn>,
    DeviceTypeCase<DespawnType, DespawnType::tracing, TracingDespawn>>;

struct DespawnVectorVisitor {
    const atlas::Geometry& query;
    const Float3& vector;
    float value;
    template <typename Tag>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(Tag) const noexcept {
        using Op = typename Tag::type;
        return Op::despawn(query, Float3(0.0f, 0.0f, 0.0f), vector, value);
    }
};
struct DespawnPositionVisitor {
    const atlas::Geometry& query;
    const Float3& position;
    const Float3& vector;
    float value;
    template <typename Tag>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(Tag) const noexcept {
        using Op = typename Tag::type;
        return Op::despawn(query, position, vector, value);
    }
};

struct Despawn final {

    DespawnType type = DespawnType::surface;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Despawn() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Despawn(DespawnType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Despawn(const Despawn& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Despawn&
    operator=(const Despawn& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~Despawn() noexcept = default;

    template <typename Payload,
              std::enable_if_t<DespawnTypeSwitch::holds<std::decay_t<Payload>>, int> = 0>
    ATLAS_HOST
    Despawn(const Payload&) noexcept
        : type(DespawnTypeSwitch::tag_of<std::decay_t<Payload>>()) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3& vector,
            float value = 0.0f) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3& position,
            const Float3& vector,
            float value = 0.0f) const noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Despawn::Despawn(const DespawnType type) noexcept
    : type(type) {
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Despawn::despawn(const atlas::Geometry& query,
                 const Float3& vector,
                 const float value) const noexcept {
    return DespawnTypeSwitch::visit(
        type,
        DespawnVectorVisitor { query, vector, value },
        false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Despawn::despawn(const atlas::Geometry& query,
                 const Float3& position,
                 const Float3& vector,
                 const float value) const noexcept {
    return DespawnTypeSwitch::visit(
        type,
        DespawnPositionVisitor { query, position, vector, value },
        false);
}

}
