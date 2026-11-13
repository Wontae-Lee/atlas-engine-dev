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

A `Unit` owns its geometry, pose, and motion state by value:

- **`Geometry _geometry`** — the collision shape in its own local frame. It is
  never modified by motion; all movement is carried by the pose.
- **`Sync _sync`** — the local-to-world pose (translation + orientation, plus
  `Sync`'s cached matrices).
- **Four `std::optional<Float3>` kinematic fields** — `_velocity` (units/s),
  `_acceleration` (units/s²), `_angular_velocity` (axis scaled by rad/s), and
  `_angular_acceleration` (axis scaled by rad/s²). An empty optional means "this
  channel is absent"; a body with no velocity is *static* and `update` is a no-op
  for it. `dynamic()` reports whether any linear or angular velocity is present.
- **Optional angular oscillation** — a normalized world-space axis, signed
  amplitude, positive period, bounded double-precision elapsed phase, current
  prescribed angle. Its analytic derivatives populate the ordinary
  angular velocity and acceleration fields.

The four kinematic fields are kept in a canonical pairing by
`canonicalize_kinematics`: if an acceleration is present without its matching
velocity (or a velocity without its acceleration), the missing partner is
defaulted to zero. This lets `update` read acceleration only through its paired
velocity and lets `dynamic()` decide motion from velocity presence alone.

## Integrating the pose — `update(dt)`

`update(float dt)` advances ordinary kinematics by one step using
**semi-implicit (symplectic) Euler**, and does nothing when `dt` is non-positive
or non-finite. Without an angular oscillation, the integration is:

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
  rotation), renormalizes to counter drift, and updates the cached matrices via
  `Sync::set_orientation()`. A zero-length axis is a no-op. Oscillation uses this
  same rotation path and does not change its phase when a caller rotates the pose.

## Prescribed angular oscillation

Use `Unit::Builder::with_angular_oscillation(axis, amplitude, period)` for a
smooth closing/opening cycle. The axis is expressed in world space and passes
through the Unit's translation; the geometry must be centered on that pivot in
its local coordinates. The amplitude is the signed angle in radians reached at
half a cycle, and the period is the duration of a complete cycle in seconds.
The initial pose is the open/rest pose:

```text
theta(t) = amplitude / 2 * (1 - cos(2*pi*t/period))
omega(t) = amplitude * pi / period * sin(2*pi*t/period)
alpha(t) = 2 * amplitude * pi^2 / period^2 * cos(2*pi*t/period)
```

The motion starts at zero angular velocity, reaches the specified amplitude at
half a period with zero angular velocity again, and returns to its initial pose
at a full period. A negative amplitude reverses the closing direction. Linear
translation/acceleration may be combined with it, so the pivot can also move.

`update()` evaluates the prescribed angle, subtracts its previous value, and
passes that increment to the existing `Unit::rotate()`. It reduces double-precision
elapsed time modulo the period, avoiding unbounded time growth. Rotation retains
the ordinary quaternion normalization and floating-point roundoff of `rotate()`;
there is no separate quaternion composition or pose-rebasing implementation.
Phase calculations use `pi_double` from `math/constants.h`.
The angular velocity and acceleration accessors report the analytic
world-space derivatives at the current phase. Collision responses therefore use
the same motion as the rendered boundary. As with other moving colliders,
particle intersections use the boundary pose at the current step; use a small
time step relative to the motion period.

```cpp
Unit wing = Unit::builder()
    .with_geometry(hinge_centered_geometry)
    .with_sync(hinge_pose)
    .with_angular_oscillation(Float3(1.0f, 0.0f, 0.0f), 0.35f, 4.0f)
    .build();
```

The Core builder rejects a non-finite or zero axis, non-finite amplitude,
non-finite/non-positive period, angular derivatives outside the finite float
range, and combinations with explicit angular velocity or acceleration. Axis
normalization scales by the largest absolute component before calling the
existing `normalized_or()` helper, avoiding squared-length overflow or underflow
for very large or small finite axes. Zero amplitude is accepted.

`set_sync()` and `rotate()` retain their ordinary pose-editing behavior. They
leave the oscillation phase, axis, and derivatives unchanged; the next update
applies only the next angular increment to the edited pose. `move()` translates
the pivot normally. Rebuilding the Unit starts a new cycle at rest. The motion
remains a trivially copyable value inside Unit and works through the same
TBB/CUDA update path.

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
members and the optional angular-oscillation state contain trivially copyable
types. This is a hard requirement,
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
