# 7. Extensibility Patterns

This document describes the extension points of `include/atlas/` — the places a
programmer adds a new strategy — and the pattern each one uses. The goal is that
adding a new case touches a small, predictable set of locations and that a
forgotten case fails to compile rather than failing silently at runtime.

There are three patterns. Pick the one that matches where the new behavior runs.

---

## 7.1 The Three Patterns

```text
Pattern A — Virtual interface (open set, host-side)
  Base class with virtual methods; add a case by subclassing.
  Used by: Solver<T>, Codec<T>, Searcher<T>, Measurer<T>.

Pattern B — DeviceVariant operator union (closed set, device-callable)
  A tagged union dispatched through DeviceVariant; add a case by editing the
  case list. Used by: GeometryOperator, GenerateOperator,
  SurfaceInteractionKernel, PostColliderKernel, DsmcKernel, SphKernel.

Pattern C — Tag-only static dispatch (closed set, stateless)
  An enum tag that selects a static function; no per-case payload stored.
  Used by: SpawnOperator, DespawnOperator.
```

Pattern A is for host-side objects assembled at setup time (a new solver, a new
searcher). Patterns B and C are for value types that must run inside device
kernels, where virtual dispatch is unavailable.

---

## 7.2 Pattern A: Virtual Interface (Open)

When the strategy runs on the host and is chosen at setup time, derive from the
abstract base and override its methods. No enum, no central dispatcher, and no
existing file needs editing — this already satisfies the open/closed principle.

```text
To add a new solver:
  1. class MySolver<T> : public Solver<T> { ... override solve(...) ... };
  2. Install it like any other solver (Orchestrator builder / with_solver).
```

The same recipe applies to `Codec<T>`, `Searcher<T>`, and `Measurer<T>`.

---

## 7.3 Pattern B: DeviceVariant + `visit` (Closed, Device-Callable)

A device-callable value type cannot use virtual dispatch, so it stores a tagged
union of the concrete payloads and dispatches on the tag. The shared helper
`atlas::detail::DeviceVariant` (in `core/detail/device_variant.h`) owns the
case list and provides lifetime management (`construct`, `copy_construct`,
`assign`, `destroy`) and behavior dispatch (`visit`).

`GeometryOperator<T>` is the reference implementation. Read
`geometry/geometry_operator.h` and `.hpp` alongside this section.

### The single source of truth

The case list exists **once**, as the `DeviceVariantCase` entries in the
variant alias:

```cpp
template <typename T>
using GeometryOperatorVariant = DeviceVariant<
    GeometryOperator<T>, GeometryType, GeometryType::Sphere,
    DeviceVariantCase<GeometryOperator<T>, GeometryType, GeometryType::Box,
                      BoxGeometryOperator<T>, &GeometryOperator<T>::box>,
    /* ...one DeviceVariantCase per shape... */>;
```

### Behavior methods are one line each

Each behavior method dispatches through `visit`, which calls the visitor on the
active payload and returns a caller-supplied fallback if the tag matches no case
(an unreachable path in practice, kept for the exact semantics of the previous
`default:` arm):

```cpp
template <typename T>
T GeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.signed_distance(p); },
        std::numeric_limits<T>::infinity());
}
```

The visitor is a generic `ATLAS_ALL_DEVICE` lambda so it works under both
backends (it requires CUDA `--extended-lambda`, which the build already
enables). Because the visitor calls a method that every payload provides, the
behavior method never enumerates the cases — the case list stays in one place.

### Recipe: add a new payload to a DeviceVariant union

```text
1. Add the enum case            (e.g. GeometryType)
2. Add the union member         (the *.h operator struct)
3. Add the explicit constructor (decl in .h, one-line def in .hpp)
4. Add one DeviceVariantCase    (the variant alias in .hpp)
-- behavior methods need NO change; visit picks the case up automatically --
```

Contrast this with the old shape, where every behavior method carried its own
`switch (type)`: adding one payload meant editing ~9 switch statements, and a
forgotten arm became a silent wrong result via `default:`. With `visit`, a
payload that does not provide the expected method fails to compile.

### DeviceVariant dispatch primitives

`DeviceVariant` provides three dispatch primitives. Pick the one that matches
the behavior method's shape:

```text
visit(owner, visitor, fallback)      value-returning, reads the active payload
                                     (const owner); returns fallback if the tag
                                     matches no case.
apply(owner, visitor)                void; const and non-const owner overloads.
                                     Use for methods that return void (including
                                     out-parameter and mutating methods).
visit_type(tag, visitor, fallback)   static type dispatch with no instance. The
                                     visitor receives detail::type_tag<Payload>
                                     and recovers the payload type via
                                     decltype(tag)::type to call a static method.
```

Invariant for `visit`/`apply`: `owner.type` always equals the active member's
tag (maintained by construct/copy/assign), so the visitor is always invoked on
the active payload; only a corrupted tag reaches the fallback (or, for `apply`,
does nothing).

### When a method is not uniform

`visit`/`apply` require the visitor to call the **same expression** on every
payload. A behavior method whose cases differ — a payload with a different
argument list, or a case that returns a constant instead of calling the payload —
is not uniform and is left as an explicit `switch`. Forcing it into a visitor
would require changing payload interfaces, which is a separate decision. The
remaining switches in 7.5 are these non-uniform cases.

---

## 7.4 Pattern C: Tag-Only Static Dispatch (Closed, Stateless)

`SpawnOperator` and `DespawnOperator` store only an enum tag and forward to a
static method on a separate stateless type. They carry no payload union.

```text
To add a new spawn/despawn type:
  1. Add the enum case (SpawnType / DespawnType)
  2. Add a stateless operator type with the required static method(s)
  3. Add the explicit constructor
  4. Add the case to the dispatch site(s)
```

These could be migrated to Pattern B for consistency, but they are stateless and
the migration is optional. If migrated, follow the `visit` recipe above.

---

## 7.5 Rollout Status

Migration of the Pattern B unions to the dispatch primitives. "Uniform" methods
are migrated; non-uniform methods are listed as the exceptions that stay as
`switch`.

```text
[done]    geometry/geometry_operator        all 9 behavior methods -> visit
[done]    collider/kernel/post_collider_kernel  sweep_motion, operator() -> apply
[done]    solver/sph/sph_kernel             3 static dispatchers -> visit_type
[partial] collider/interaction/surface_interaction_kernel
            operator() -> visit
            internal_energy stays (isothermal returns the input unchanged)
[partial] generator/generate_operator
            reseed -> apply
            generate() x2 stay (maxwell_sigma takes one fewer argument)
[partial] solver/dsmc/dsmc_kernel
            operator() -> apply
            cross_section stays (hard_sphere has no relative_speed argument)
[optional] source/spawn_operator, sink/despawn_operator  (Pattern C -> B)
```

The `[partial]` exceptions are non-uniform behavior methods (see 7.3). Making
them uniform would mean changing the payload interfaces (for example giving every
generator a two-argument `generate`); that is deliberately out of scope here and
should be decided on its own merits.

When migrating, keep the change behavior-preserving: replace each uniform
`switch (type)` method with a `visit`/`apply`/`visit_type` call whose fallback
equals the old `default:` value, and leave the lifetime/case-list code untouched.
