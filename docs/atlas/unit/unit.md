# Unit

A `Unit` is the world-space instance of a rigid body: a collision `Geometry`
(expressed in a fixed local frame) coupled with its local-to-world pose (a
`Sync`) and an optional first-order kinematic state (linear and angular velocity
and acceleration). It is **not** a units-of-measurement type — it is the
pose/kinematics object a collider leaf owns. `IsothermalCollider` holds one by
value and drives it: `trace` ray-casts against its geometry, `surface_velocity`
supplies the moving wall's speed at a hit, and `advance` steps its motion. Every
accessor and the integrator are `__host__ __device__`, and the whole object is
trivially copyable, so a `Unit` rides inside a `DeviceBuffer<Collider>` and is
captured by value into device lambdas.

## Files

| File | Role |
|---|---|
| `include/atlas/unit/unit.h` | `Unit` (pose + kinematics + owned `Geometry`) and its host-only `Unit::Builder` |
| `src/atlas/unit/unit.cu` | `Unit::Builder` implementation: `with_*` setters, `validate`, `build`, `make_host_shared` |

## State it carries

A `Unit` owns six members by value:

- **`Geometry _geometry`** — the collision shape in its own local frame. It is
  never modified by motion; all movement is carried by the pose.
- **`Sync _sync`** — the local-to-world pose (translation + orientation, plus
  `Sync`'s cached matrices).
- **Four `std::optional<Float3>` kinematic fields** — `_velocity` (units/s),
  `_acceleration` (units/s²), `_angular_velocity` (axis scaled by rad/s), and
  `_angular_acceleration` (axis scaled by rad/s²). An empty optional means "this
  channel is absent"; a body with no velocity is *static* and `update` is a no-op
  for it. `dynamic()` reports whether any linear or angular velocity is present.

The four kinematic fields are kept in a canonical pairing by
`canonicalize_kinematics`: if an acceleration is present without its matching
velocity (or a velocity without its acceleration), the missing partner is
defaulted to zero. This lets `update` read acceleration only through its paired
velocity and lets `dynamic()` decide motion from velocity presence alone.

## Integrating the pose — `update(dt)`

`update(float dt)` advances the pose by one step using **semi-implicit
(symplectic) Euler**, and does nothing when `dt` is non-positive (the guard is
`if (!(dt > 0.0f)) return;`, which also rejects NaN):

```cpp
if (_velocity.has_value()) {
    if (_acceleration.has_value()) *_velocity += (*_acceleration) * dt; // v += a·dt first
    move((*_velocity) * dt);                                            // then displace by v·dt
}

if (_angular_velocity.has_value()) {
    if (_angular_acceleration.has_value()) *_angular_velocity += (*_angular_acceleration) * dt;
    const float omega = _angular_velocity->length();
    if (omega > 0.0f) rotate(*_angular_velocity, omega * dt);           // magnitude = rate, direction = axis
}
```

Acceleration is integrated into velocity first, then the *updated* velocity
displaces the pose. Linear and angular channels are independent, and each is
skipped when its velocity is absent. For rotation, the angular velocity's
magnitude is the rate and its direction the axis; a zero-magnitude angular
velocity applies no rotation, avoiding a normalize of a zero axis.

Two lower-level mutators back `update` and are also public:

- **`move(delta)`** adds `delta` directly to `_sync.translation`; orientation and
  the cached matrices are untouched.
- **`rotate(axis, angle_rad)`** left-multiplies an incremental
  `Quaternion::from_axis_angle` onto the current orientation (world-frame
  rotation), renormalizes to counter drift, and calls `_sync.rebuild_matrices()`.
  A zero-length axis is a no-op.

## Surface velocity — `surface_velocity(point)`

`surface_velocity(surface_point)` returns the instantaneous world-space velocity
of a material point, as the rigid-body sum **v + ω × r**:

```cpp
Float3 velocity(0.0f, 0.0f, 0.0f);
if (_velocity.has_value())         velocity += *_velocity;
if (_angular_velocity.has_value()) velocity += atlas::cross(*_angular_velocity, surface_point - _sync.translation);
return velocity;
```

Here `r = surface_point - _sync.translation` is the vector from the body origin
(the pose translation) to the query point. Absent components contribute zero, so
a static body yields the zero vector. `IsothermalCollider::collide` uses this to
give a reflected particle the moving wall's local surface speed
(`velocity - wall_velocity`, reflect, then `+ wall_velocity`).

## World-space queries and why there is no cache to invalidate

The geometry is stored **once, in its local frame**, and every world-space query
transforms through the pose on demand rather than maintaining a separate
world-space copy of the geometry:

- **`trace(world_ray)`** pulls the ray into the local frame
  (`_sync.sync_to_local`), intersects the *local* geometry, and — only on a hit —
  pushes the point and normal back out (`sync_to_world` for the point,
  `sync_dir_to_world` for the normal, so it stays a pure rotation). The reported
  hit distance is a valid world distance because a rigid pose preserves lengths.
- **`world_bound()`** transforms all eight corners of the geometry's local AABB
  into world space and merges them into a fresh AABB — a conservative but always
  valid bound of the posed (possibly rotated) box. If the local geometry has no
  valid bound (e.g. an infinite plane), an empty AABB is returned.

Because motion only ever mutates `_sync` (through `move`/`rotate`) and never the
geometry, advancing a unit leaves its local geometry unchanged, so **there is no
world-space geometry cache on the `Unit` to invalidate** — this confirms the
statement in [collider.md](../collider/collider.md) that "the local geometry is
unchanged so no cache invalidation is needed." (The *collider* leaf does cache a
world-space `AABB`, but that is the collider's own broad-phase cache, refreshed
in `IsothermalCollider::advance` after `_unit.update(dt)`; the `Unit` itself
holds nothing derived that could go stale.)

## Trivial copyability

`Unit` is trivially copyable: `Geometry` is a `DeviceVariant` (trivially
copyable), `Sync` is a plain value type, and the four `std::optional<Float3>`
members are optionals of a trivially copyable type. This is a hard requirement,
not an accident — `Collider` leaves must be trivially copyable to live in a
`DeviceVariant` union and be captured into device kernels, and each leaf owns a
`Unit` by value. The class-level Doxygen phrases this as "trivially movable"; the
type is in fact trivially copyable, which is the stronger property the collider
relies on.

## Building a Unit — `Unit::Builder`

`Unit::builder()` returns a host-only fluent builder. Geometry and sync are
required; the four kinematic fields are optional.

```cpp
Unit unit = Unit::builder()
    .with_geometry(box_geometry)                 // required
    .with_sync(sync_host_ptr)                     // required, non-null (SyncHostPtr, deep-copied)
    .with_velocity(Float3(0.0f, 0.0f, 2.0f))      // optional
    .with_acceleration(Float3(0.0f, 0.0f, -9.8f)) // optional
    .with_angular_velocity(Float3(0.0f, 0.0f, 1.0f))
    .with_angular_acceleration(Float3(0.0f, 0.0f, 0.5f))
    .build();
```

- **`with_sync`** takes a `SyncHostPtr` and stores a *value copy* of the pointee,
  so the built `Unit` owns its pose independently of the shared handle; a null
  handle throws.
- **`build()`** runs `validate()`, canonicalizes the kinematics on local copies,
  moves everything into a bare `Unit` (via friendship, bypassing the value
  constructor since the kinematics are already canonical), then **resets the
  builder's staged state** — it is single-use and must be reconfigured before a
  second `build()`.
- **`make_host_shared()`** builds and wraps the result in a
  `atlas::host_shared_ptr<Unit>`.

`validate()` throws `std::runtime_error` when geometry or sync is unset, when a
linear acceleration is set without a linear velocity, or when an angular
acceleration is set without an angular velocity. Note the asymmetry with the
value constructor: `canonicalize_kinematics` *would* zero-fill a missing velocity
partner, but through the builder `validate()` rejects that case first, so you must
supply a velocity alongside any acceleration (as the `with_acceleration` Doxygen
note states). Canonicalization in the builder path therefore only ever fills the
reverse direction (a velocity without an acceleration → zero acceleration).

## Related documentation

- [geometry.md](../geometry/geometry.md) — the collision shape a `Unit` owns and
  traces against in its local frame.
- [collider.md](../collider/collider.md) — how `IsothermalCollider` owns a `Unit`
  and calls `trace`, `surface_velocity`, and `update`/`advance`.
