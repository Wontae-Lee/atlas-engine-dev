#pragma once

#include <atlas/collider/collider_type.h>
#include <atlas/collider/isothermal_collider.h>
#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/ray.h>

#include <concepts>
#include <type_traits>

namespace atlas {

/**
 * @brief Compile-time contract every `Collider` leaf must satisfy.
 *
 * A leaf models a collidable moving wall: it can be swept-tested against a
 * particle ray (`trace`), resolve the resulting hit (`collide`), advance its own
 * motion (`advance`), and expose a cached world bound for broad-phase culling
 * (`bound`). Enforcing the shape as a concept lets `Collider`'s dispatch and the
 * `static_assert` below fail loudly at compile time if a new leaf drifts from
 * the expected signatures.
 *
 * @tparam C Candidate leaf type.
 */
template <typename C>
concept ConceptCollider = requires(C collider, const HitSurface hit, Float3 vec, float dt) {
    { collider.trace(vec, vec, dt) } -> std::same_as<HitSurface>;  ///< Narrow-phase sweep → hit.
    { collider.collide(hit, vec, vec, dt) } -> std::same_as<void>; ///< Apply a hit in place.
    { collider.bound() } -> std::same_as<const AABB&>;             ///< Cached world AABB.
    { collider.advance(dt) } -> std::same_as<void>;                ///< Step the wall's motion.
};

/// Guarantees the only leaf conforms to the contract; add one per new leaf.
static_assert(ConceptCollider<IsothermalCollider>);

/**
 * @brief Tagged-union umbrella over the collider leaf types.
 *
 * Holds a `ColliderType` tag plus a `union` of leaves and forwards every
 * operation to the active leaf through `ColliderVariant` (a `DeviceVariant`).
 * Because every leaf is trivially copyable, `Collider` is too, so it lives in a
 * `DeviceBuffer<Collider>` and dispatches on the device — the whole point of
 * using `DeviceVariant` here rather than the host-only `HostVariant`.
 *
 * The public data members (`type`, the anonymous `union`) are intentionally
 * exposed: `DeviceVariant` addresses the union member through a
 * pointer-to-member (`DeviceVariantCase`) and reads `type` directly, both of
 * which require public access.
 */
class Collider final {
public:
    ColliderType type = ColliderType::isothermal; ///< Active-leaf discriminator for the union.

    /// The leaf storage; exactly the member named by `type` is alive at a time.
    union {

        IsothermalCollider isothermal; ///< Live when `type == ColliderType::isothermal`.
    };

    /**
     * @brief Default-constructs the collider with its default leaf active.
     *
     * Constructs the `isothermal` union member via `ColliderVariant::construct`,
     * matching `type`'s default. Needed so `Collider` itself is default
     * constructible (e.g. as a buffer element).
     */
    ATLAS_ALL_DEVICE
    Collider() noexcept;

    /**
     * @brief Trivial copy: valid because every leaf is trivially copyable.
     * @param other Source collider to copy.
     */
    ATLAS_ALL_DEVICE
    Collider(const Collider& other) noexcept = default;

    /**
     * @brief Trivial copy assignment; see the copy constructor.
     * @param other Source collider to copy.
     * @return `*this`.
     */
    ATLAS_ALL_DEVICE Collider&
    operator=(const Collider& other) noexcept = default;

    /// Trivial destructor; leaves own no heap resources requiring teardown.
    ATLAS_ALL_DEVICE
    ~Collider() noexcept = default;

    /**
     * @brief Constructs from a concrete leaf, setting `type` and placing it in the union.
     *
     * Delegates to `ColliderVariant::construct_payload`, which infers the tag
     * from the leaf's type and copy-constructs it into the matching union
     * member. The SFINAE guard excludes `Collider` itself so this does not
     * shadow the copy constructor.
     *
     * @tparam Payload A leaf type (e.g. `IsothermalCollider`), never `Collider`.
     * @param collider The leaf to wrap; copied into the union.
     */
    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Collider>, int> = 0>
    ATLAS_ALL_DEVICE explicit Collider(const Payload& collider) noexcept;

    /**
     * @brief Advances the active leaf's wall motion by `dt` (mutable dispatch).
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(float dt) noexcept;

    /**
     * @brief Cached world AABB of the active leaf, for broad-phase culling.
     * @return Const reference to the leaf's bound (invalid for unbounded geometry).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB&
    bound() const noexcept;

    /**
     * @brief Narrow-phase sweep of the particle path against the active leaf.
     * @param position World-space start position of the step.
     * @param velocity World-space particle velocity.
     * @param dt Time step in seconds.
     * @return The leaf's `HitSurface` (non-intersecting on a miss).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Float3& position, const Float3& velocity, float dt) const noexcept;

    /**
     * @brief Resolves a hit against the active leaf, updating position/velocity.
     * @param hit The hit chosen by the caller (from `trace`).
     * @param position In/out world-space position.
     * @param velocity In/out world-space velocity.
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    collide(const HitSurface& hit, Float3& position, Float3& velocity, float dt) const noexcept;
};

/**
 * @brief `DeviceVariant` specialization that drives `Collider`'s dispatch.
 *
 * Binds the owner type, its tag enum, the default tag, and one
 * `DeviceVariantCase` per leaf (mapping a `ColliderType` value to the union
 * member pointer). All of `Collider`'s methods route through this alias's
 * static `construct`/`visit`/`apply` helpers.
 */
using ColliderVariant = DeviceVariant<
    Collider,
    ColliderType,
    ColliderType::isothermal,
    DeviceVariantCase<ColliderType::isothermal, &Collider::isothermal>>;

/**
 * @brief `DeviceVariant` visitor forwarding `advance(dt)` to the active leaf.
 *
 * A stateless-but-for-`dt` function object; `DeviceVariant::apply` calls it with
 * the live leaf by reference so it can mutate the wall's motion.
 */
class ColliderAdvance {
public:
    float dt; ///< Time step in seconds, forwarded to `leaf.advance`.
    /**
     * @brief Calls `advance(dt)` on the active leaf.
     * @tparam C Deduced leaf type.
     * @param collider The live leaf, mutated in place.
     */
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(C& collider) const noexcept { collider.advance(dt); }
};

/**
 * @brief `DeviceVariant` visitor returning a pointer to the leaf's cached bound.
 *
 * Returns a pointer (not a reference) because `DeviceVariant::visit` needs a
 * copyable, default-constructible fallback value; `Collider::bound` dereferences
 * the result.
 */
class ColliderBound {
public:
    /**
     * @brief Returns the address of the active leaf's cached world AABB.
     * @tparam C Deduced leaf type.
     * @param collider The live leaf.
     * @return Pointer to the leaf's bound.
     */
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB*
    operator()(const C& collider) const noexcept { return &collider.bound(); }
};

/**
 * @brief `DeviceVariant` visitor forwarding `trace(position, velocity, dt)`.
 *
 * Captures the sweep inputs by reference/value and returns the leaf's hit so
 * `DeviceVariant::visit` can produce a `HitSurface` from the active leaf.
 */
class ColliderTrace {
public:
    const Float3& position; ///< World-space start position of the step.
    const Float3& velocity; ///< World-space particle velocity.
    float dt;               ///< Time step in seconds.
    /**
     * @brief Calls `trace` on the active leaf.
     * @tparam C Deduced leaf type.
     * @param collider The live leaf.
     * @return The leaf's `HitSurface`.
     */
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const C& collider) const noexcept { return collider.trace(position, velocity, dt); }
};

/**
 * @brief `DeviceVariant` visitor forwarding `collide(hit, position, velocity, dt)`.
 *
 * Holds the hit plus in/out references to the particle state so
 * `DeviceVariant::apply` can let the active leaf write the post-collision result.
 */
class ColliderCollide {
public:
    const HitSurface& hit; ///< The chosen hit to resolve.
    Float3& position;      ///< In/out world-space position.
    Float3& velocity;      ///< In/out world-space velocity.
    float dt;              ///< Time step in seconds.
    /**
     * @brief Calls `collide` on the active leaf.
     * @tparam C Deduced leaf type.
     * @param collider The live leaf.
     */
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(const C& collider) const noexcept { collider.collide(hit, position, velocity, dt); }
};

// Out-of-line definitions: each simply routes to the matching ColliderVariant
// helper. They live here (not in a .cu) because Collider is header-only and
// ATLAS_ALL_DEVICE, so the device compiler must see the bodies.

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Collider::Collider() noexcept {
    // Bring the default leaf to life so the union has a live member from the start.
    ColliderVariant::construct(*this, ColliderType::isothermal);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Collider>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Collider::Collider(const Payload& collider) noexcept {
    ColliderVariant::construct_payload(*this, collider);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
Collider::advance(const float dt) noexcept {
    ColliderVariant::apply(*this, ColliderAdvance { dt });
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB&
Collider::bound() const noexcept {
    // ColliderBound returns a pointer; the null fallback can only surface for an
    // unrecognized tag, which the leaf invariant rules out, so the deref is safe.
    return *ColliderVariant::visit(
        *this,
        ColliderBound {},
        static_cast<const AABB*>(nullptr));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
Collider::trace(const Float3& position, const Float3& velocity, const float dt) const noexcept {
    return ColliderVariant::visit(
        *this,
        ColliderTrace { position, velocity, dt },
        HitSurface {});
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
Collider::collide(const HitSurface& hit, Float3& position, Float3& velocity, const float dt) const noexcept {
    ColliderVariant::apply(
        *this,
        ColliderCollide { hit, position, velocity, dt });
}

/// Host-side shared-ownership handle to a `Collider` umbrella.
using ColliderHostPtr = atlas::host_shared_ptr<Collider>;

/// Device-side shared-ownership handle to a `Collider` umbrella.
using ColliderDevicePtr = atlas::device_shared_ptr<Collider>;

}