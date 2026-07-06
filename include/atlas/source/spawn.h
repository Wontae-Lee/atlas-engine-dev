#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>

#include <type_traits>

namespace atlas {

enum class SpawnType : int {

    surface,

    volume
};

struct SurfaceSpawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    spawn(const atlas::Geometry& query,
          const Float3& particle,
          const float tolerance = 0.0f) noexcept {
        return query.is_on_surface(particle, tolerance);
    }
};

struct VolumeSpawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    spawn(const atlas::Geometry& query,
          const Float3& particle,
          const float tolerance = 0.0f) noexcept {
        return query.is_inside(particle, tolerance);
    }
};

using SpawnTypeSwitch = DeviceTypeSwitch<
    SpawnType,
    SpawnType::surface,
    DeviceTypeCase<SpawnType, SpawnType::surface, SurfaceSpawn>,
    DeviceTypeCase<SpawnType, SpawnType::volume, VolumeSpawn>>;

struct SpawnVisitor {
    const atlas::Geometry& query;
    const Float3& particle;
    float tolerance;
    template <typename Tag>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator()(Tag) const noexcept {
        using Op = typename Tag::type;
        return Op::spawn(query, particle, tolerance);
    }
};

struct Spawn final {

    SpawnType type = SpawnType::surface;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Spawn() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Spawn(SpawnType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Spawn(const Spawn& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Spawn&
    operator=(const Spawn& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~Spawn() noexcept = default;

    template <typename Payload,
              std::enable_if_t<SpawnTypeSwitch::holds<std::decay_t<Payload>>, int> = 0>
    ATLAS_HOST
    Spawn(const Payload&) noexcept
        : type(SpawnTypeSwitch::tag_of<std::decay_t<Payload>>()) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    spawn(const atlas::Geometry& query,
          const Float3& particle,
          float tolerance = 0.0f) const noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Spawn::Spawn(const SpawnType type) noexcept
    : type(type) {
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Spawn::spawn(const atlas::Geometry& query,
             const Float3& particle,
             const float tolerance) const noexcept {

    return SpawnTypeSwitch::visit(
        type,
        SpawnVisitor { query, particle, tolerance },
        false);
}

}
