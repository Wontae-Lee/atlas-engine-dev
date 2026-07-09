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

template <typename C>
concept ConceptCollider = requires(C collider, const HitSurface hit, Float3 vec, float dt) {
    { collider.trace(vec, vec, dt) } -> std::same_as<HitSurface>;
    { collider.collide(hit, vec, vec, dt) } -> std::same_as<void>;
    { collider.bound() } -> std::same_as<const AABB&>;
    { collider.advance(dt) } -> std::same_as<void>;
};

static_assert(ConceptCollider<IsothermalCollider>);

class Collider final {
public:
    ColliderType type = ColliderType::isothermal;

    union {

        IsothermalCollider isothermal;
    };

    ATLAS_ALL_DEVICE
    Collider() noexcept;

    ATLAS_ALL_DEVICE
    Collider(const Collider& other) noexcept = default;

    ATLAS_ALL_DEVICE Collider&
    operator=(const Collider& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~Collider() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, Collider>, int> = 0>
    ATLAS_ALL_DEVICE explicit Collider(const Payload& collider) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    advance(float dt) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB&
    bound() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Float3& position, const Float3& velocity, float dt) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    collide(const HitSurface& hit, Float3& position, Float3& velocity, float dt) const noexcept;
};

using ColliderVariant = DeviceVariant<
    Collider,
    ColliderType,
    ColliderType::isothermal,
    DeviceVariantCase<ColliderType::isothermal, &Collider::isothermal>>;

class ColliderAdvance {
public:
    float dt;
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(C& collider) const noexcept { collider.advance(dt); }
};

class ColliderBound {
public:
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const AABB*
    operator()(const C& collider) const noexcept { return &collider.bound(); }
};

class ColliderTrace {
public:
    const Float3& position;
    const Float3& velocity;
    float dt;
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const C& collider) const noexcept { return collider.trace(position, velocity, dt); }
};

class ColliderCollide {
public:
    const HitSurface& hit;
    Float3& position;
    Float3& velocity;
    float dt;
    template <typename C>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(const C& collider) const noexcept { collider.collide(hit, position, velocity, dt); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Collider::Collider() noexcept {
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

using ColliderHostPtr   = atlas::host_shared_ptr<Collider>;
using ColliderDevicePtr = atlas::device_shared_ptr<Collider>;

}