#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/sink/surface_despawn.h>
#include <atlas/sink/tracing_despawn.h>
#include <atlas/sink/volume_despawn.h>

#include <type_traits>

/**
 * @file despawn.h
 * @brief Runtime-selectable dispatcher over the three despawn rules
 *        (`SurfaceDespawn`, `VolumeDespawn`,
 *        `TracingDespawn`) a `Sink`'s units can use.
 *
 * @details
 * Follows the tag-dispatch `DeviceTypeSwitch` pattern (a lighter-weight
 * relative of `DeviceVariant` used elsewhere in Atlas — here the
 * operators are all stateless, so there is no payload to store, only a
 * type tag to switch on and call the matching operator's static
 * `despawn`): `DespawnType` selects which operator's `despawn()` a
 * `Despawn` value forwards to, dispatched via
 * `detail::DespawnTypeSwitch::visit`. Two overloads exist because
 * `TracingDespawn` needs both the particle's position *and* its
 * velocity to build a sweep, while `SurfaceDespawn`/
 * `VolumeDespawn` only need the local position (the 3-argument
 * overload's leading `Vector3(0,0,0)` is an unused placeholder position
 * for those two).
 */

namespace atlas {

/**
 * @brief Selects which despawn rule a `Despawn` applies; see
 *        `surface_despawn.h`, `volume_despawn.h`,
 *        `tracing_despawn.h` for each rule's condition.
 */
enum class DespawnType : int {

    /** Particle touches the unit's surface (`SurfaceDespawn`). */
    surface,

    /** Particle is inside the unit's volume (`VolumeDespawn`). */
    volume,

    /** Particle's swept motion this timestep crosses the unit's surface
     *  (`TracingDespawn`). */
    tracing
};

namespace detail {

    using DespawnTypeSwitch = DeviceTypeSwitch<
        DespawnType,
        DespawnType::surface,
        DeviceTypeCase<DespawnType, DespawnType::surface, SurfaceDespawn>,
        DeviceTypeCase<DespawnType, DespawnType::volume, VolumeDespawn>,
        DeviceTypeCase<DespawnType, DespawnType::tracing, TracingDespawn>>;

    // Functor visitors instead of generic device lambdas (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas).
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

}

/**
 * @brief Tag-dispatch wrapper letting `Sink` call `despawn()` on
 *        whichever `DespawnType` a unit was configured with, without
 *        virtual dispatch or a payload union (every rule is stateless).
 *        See this file's top-of-file documentation for the
 *        `DeviceTypeSwitch` pattern.
 */
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
              std::enable_if_t<detail::DespawnTypeSwitch::holds<std::decay_t<Payload>>, int> = 0>
    ATLAS_HOST
    Despawn(const Payload&) noexcept
        : type(detail::DespawnTypeSwitch::tag_of<std::decay_t<Payload>>()) {
    }

    /** @brief Position-only despawn test (`Surface`/`Volume` rules);
     *  `vector` is the local particle position, `value` the tolerance. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    despawn(const atlas::Geometry& query,
            const Float3& vector,
            float value = 0.0f) const noexcept;

    /** @brief Position-and-motion despawn test (needed by `Tracing`;
     *  also usable by the position-only rules, which ignore
     *  `position`); `vector` is the local velocity for `Tracing` or
     *  local position otherwise, `value` the tolerance or timestep
     *  respectively. */
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
    return detail::DespawnTypeSwitch::visit(
        type,
        detail::DespawnVectorVisitor { query, vector, value },
        false);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
Despawn::despawn(const atlas::Geometry& query,
                 const Float3& position,
                 const Float3& vector,
                 const float value) const noexcept {
    return detail::DespawnTypeSwitch::visit(
        type,
        detail::DespawnPositionVisitor { query, position, vector, value },
        false);
}

}
