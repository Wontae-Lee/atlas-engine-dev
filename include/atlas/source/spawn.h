#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>

#include <type_traits>

/**
 * @file spawn.h
 * @brief Candidate-position acceptance rules for particle spawning: a
 *        candidate lattice point is a valid spawn site for a unit if it
 *        lies on the unit's surface (`Surface`) or inside its volume
 *        (`Volume`). Mirrors `despawn.h`'s pattern for `Sink`,
 *        applied to inflow (`Source`) instead of outflow.
 *
 * @details
 * `Source` generates a regular lattice of candidate
 * points over each unit's bounding box (see that file); `Spawn`
 * is the geometric filter deciding which of those candidates are
 * actually valid spawn locations for that unit's shape — analogous to
 * `Despawn`'s tag-dispatch design (`SpawnType` selects
 * `SurfaceSpawn`/`VolumeSpawn` via
 * `detail::SpawnTypeSwitch`), but with a single position-only `spawn()`
 * signature (unlike despawn, spawning never needs a swept-motion
 * variant).
 */

namespace atlas {

/**
 * @brief Selects which geometric acceptance test a `Spawn`
 *        applies to candidate spawn positions.
 */
enum class SpawnType : int {

    /** Candidate must lie on the unit's surface. */
    surface,

    /** Candidate must lie inside the unit's volume. */
    volume
};

/** @brief `spawn()` true iff the candidate is on the unit's surface
 *  within `tolerance` (`Geometry::is_on_surface`). */
struct SurfaceSpawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    spawn(const atlas::Geometry& query,
          const Float3& particle,
          const float tolerance = 0.0f) noexcept {
        return query.is_on_surface(particle, tolerance);
    }
};

/** @brief `spawn()` true iff the candidate is inside the unit's volume
 *  within `tolerance` (`Geometry::is_inside`). */
struct VolumeSpawn final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE bool
    spawn(const atlas::Geometry& query,
          const Float3& particle,
          const float tolerance = 0.0f) noexcept {
        return query.is_inside(particle, tolerance);
    }
};

namespace detail {

    using SpawnTypeSwitch = DeviceTypeSwitch<
        SpawnType,
        SpawnType::surface,
        DeviceTypeCase<SpawnType, SpawnType::surface, SurfaceSpawn>,
        DeviceTypeCase<SpawnType, SpawnType::volume, VolumeSpawn>>;

    // Functor visitor instead of a generic device lambda (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas).
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

}

/**
 * @brief Tag-dispatch wrapper letting `Source` test
 *        candidate positions through whichever `SpawnType` a unit was
 *        configured with. See this file's top-of-file documentation.
 */
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
              std::enable_if_t<detail::SpawnTypeSwitch::holds<std::decay_t<Payload>>, int> = 0>
    ATLAS_HOST
    Spawn(const Payload&) noexcept
        : type(detail::SpawnTypeSwitch::tag_of<std::decay_t<Payload>>()) {
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

    return detail::SpawnTypeSwitch::visit(
        type,
        detail::SpawnVisitor { query, particle, tolerance },
        false);
}

}
