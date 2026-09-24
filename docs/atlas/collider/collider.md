# Collider

The collider module resolves collisions between fluid particles and a moving
solid boundary (a `Unit`). It follows the **tagged-union leaf** pattern used
across the engine (`geometry.h`-style): a single concrete umbrella type wraps
one of several self-contained leaf types and dispatches to it.

## Files

| File | Role |
|---|---|
| `include/atlas/collider/collider.h` | `Collider` umbrella (`DeviceVariant`) + `ConceptCollider` |
| `include/atlas/collider/collider_type.h` | `enum class ColliderType` |
| `include/atlas/collider/isothermal_collider.h` | `IsothermalCollider` leaf + `Builder` |
| `include/atlas/collider/diffuse_sampling.h` | `enum class DiffuseSampling` |
| `src/atlas/collider/isothermal_collider.cu` | `IsothermalCollider::Builder` implementation |

## Design

`Collider` is a **`DeviceVariant`** (see `core/device_variant.h`): a `type` tag
plus a `union` of leaf colliders. Because every leaf is trivially copyable, the
whole `Collider` is trivially copyable and can live in a `DeviceBuffer<Collider>`
and dispatch on the device.

```cpp
class Collider {
    ColliderType type;
    union { IsothermalCollider isothermal; /* future leaves */ };
    HitSurface trace(pos, vel, dt) const;                 // visit → leaf
    void       collide(hit, pos, vel, dt) const;          // apply → leaf
    void       advance(dt);                               // apply → leaf (mutable)
};
```

### Leaf contract — `ConceptCollider`

Every leaf must satisfy `ConceptCollider`:

```cpp
template <typename C>
concept ConceptCollider = requires(C c, const HitSurface hit, Float3 vec, float dt) {
    { c.trace(vec, vec, dt) }        -> std::same_as<HitSurface>;
    { c.collide(hit, vec, vec, dt) } -> std::same_as<void>;
    { c.advance(dt) }                -> std::same_as<void>;
};
```

- **`trace(position, velocity, dt) → HitSurface`** — sweeps the particle ray
  against the leaf's own `Unit` geometry and returns the world-space hit (or an
  empty `HitSurface` on miss / out of sweep range).
- **`collide(hit, position, velocity, dt)`** — on a hit, applies the wall
  interaction (surface velocity from the unit, reflection model) and writes the
  updated `position`/`velocity`.
- **`advance(dt)`** — advances the leaf's owned `Unit` (`_unit.update(dt)`); the
  local geometry is unchanged so no cache invalidation is needed.

The per-particle loop lives **outside** the collider (the caller runs the
`parallel_for`): trace fills a per-particle `HitSurface`, collide consumes it.

### `IsothermalCollider`

The only leaf today. It owns a `Unit` by value and the reflection parameters
`momentum_accommodation_coefficient`, `restitution`, `diffuse_sampling`. Its
`reflect()` blends specular and cosine/uniform-diffuse reflection weighted by the
momentum accommodation coefficient. Surface velocity is taken from the unit via
`Unit::surface_velocity(point)`.

Constructed through `IsothermalCollider::Builder`
(`with_unit`, `with_momentum_accommodation_coefficient`, `with_restitution`,
`with_diffuse_sampling`), then wrapped: `Collider c(isothermal);`.
`Builder::validate()` requires a Unit, finite accommodation in `[0, 1]`, and
finite non-negative restitution. `build()` moves the Unit into the collider and
resets staged values. The collider owns that Unit by value; `System` stores
colliders in a backend buffer and exposes read-only pose snapshots to consumers.

## Adding a leaf

1. Write a trivially-copyable leaf satisfying `ConceptCollider` (own a `Unit`,
   implement `trace`/`collide`/`advance`, add a `Builder`).
2. Add its `enum` value to `ColliderType`.
3. Add the union member and a `DeviceVariantCase` to `ColliderVariant`.
4. Add `static_assert(ConceptCollider<NewLeaf>);`.

Because leaves are trivially copyable, `DeviceVariant` is the correct umbrella
here (contrast `Source`, which needs `HostVariant`).
