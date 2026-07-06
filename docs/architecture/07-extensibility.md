# 7. Extensibility Patterns

This document describes the extension points of `include/atlas/` — the places a
programmer adds a new strategy — and the pattern each one uses. The goal is that
adding a new case touches a small, predictable set of locations, the case list
lives in exactly one place, and a forgotten or mismatched case fails to compile
rather than failing silently at runtime.

There are two patterns. Pick the one that matches where the new behavior runs.

---

## 7.1 The Two Patterns

```text
Pattern A — Virtual interface (open set, host-side)
  Base class with virtual methods; add a case by subclassing.
  Used by: Solver, Codec, Searcher, Measurer.

Pattern B — Tagged dispatch (closed set, device-callable)
  A tag selects the active case; dispatch goes through the shared helpers in
  core/detail/device_variant.h. Two shapes share these helpers:
    - stateful operator unions (carry a payload union): GeometryOperator,
      Generate, SurfaceInteractionKernel, PostColliderKernel,
      DsmcKernel, SphKernel.
    - stateless operators (carry only the tag): Spawn, Despawn.
```

Pattern A is for host-side objects assembled at setup time (a new solver, a new
searcher). Pattern B is for value types that must run inside device kernels,
where virtual dispatch is unavailable.

No behavior method in either pattern enumerates the cases with a `switch`; the
case list lives only in the type's `DeviceVariant` / `DeviceTypeSwitch` alias.

---

## 7.2 Pattern A: Virtual Interface (Open)

When the strategy runs on the host and is chosen at setup time, derive from the
abstract base and override its methods. No enum, no central dispatcher, and no
existing file needs editing — this already satisfies the open/closed principle.

```text
To add a new solver:
  1. class MySolver : public Solver { ... override solve(...) ... };
  2. Install it like any other solver (Orchestrator builder / with_solver).
```

The same recipe applies to `Codec`, `Searcher`, and `Measurer`.

---

## 7.3 Pattern B: Tagged Dispatch (Closed, Device-Callable)

A device-callable value type cannot use virtual dispatch, so it carries a tag
and dispatches on it. The shared helpers live in `core/detail/device_variant.h`:

```text
DeviceVariant<Owner, Tag, Default, Cases...>   for stateful operator unions
  - owns a payload union and its lifetime (construct/copy_construct/assign/destroy)
  - dispatches behavior through visit / apply / visit_type

DeviceTypeSwitch<Tag, Default, Cases...>       for stateless operators
  - no payload, no lifetime management
  - dispatches a tag to a static method via visit
```

`GeometryOperator` is the reference implementation for the stateful shape;
`Spawn` is the reference for the stateless shape. Note that for
`GeometryOperator` the union members are the shape classes themselves
(`Box`, `Sphere`, ..., and the `TriangleMeshGeometryOperator` view) — each
shape is its own device value type, so there is no separate per-shape operator
struct to add; the other stateful unions (`Generate`, `DsmcKernel`,
`SphKernel`, `SurfaceInteractionKernel`) still hold dedicated payload structs.

### The single source of truth

The case list exists **once**, as the case entries in the alias. Each case is
just its tag value and the union member it owns; the owner, tag type, and payload
type are deduced from the member pointer:

```cpp
using GeometryOperatorVariant = DeviceVariant<
    GeometryOperator, GeometryType, GeometryType::Sphere,
    DeviceVariantCase<GeometryType::Box, &GeometryOperator::box>,
    /* ...one DeviceVariantCase per shape... */>;
```

For a stateless operator:

```cpp
using SpawnTypeSwitch = DeviceTypeSwitch<
    SpawnType, SpawnType::Surface,
    DeviceTypeCase<SpawnType, SpawnType::Surface, SurfaceSpawn>,
    DeviceTypeCase<SpawnType, SpawnType::Volume,  VolumeSpawn>>;
```

### Behavior methods are one line each

Each behavior method dispatches through a primitive. The case list is never
re-enumerated in the method body:

```cpp
float GeometryOperator::signed_distance(const atlas::Vector3& p) const noexcept {
    return detail::GeometryOperatorVariant::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.signed_distance(p); },
        std::numeric_limits<float>::infinity());
}
```

The visitor is a generic `ATLAS_ALL_DEVICE` lambda so it works under both
backends (it requires CUDA `--extended-lambda`, which the build already
enables). Because the visitor calls a method that every payload provides, the
method picks up new cases automatically.

### Dispatch primitives

```text
DeviceVariant:
  visit(owner, visitor, fallback)      value-returning; reads the active payload
                                       (const owner). Returns fallback only if the
                                       tag matches no case.
  apply(owner, visitor)                void; const and non-const owner overloads.
                                       Use for methods that return void, mutate
                                       the active payload, or write out-parameters.
  visit_type(tag, visitor, fallback)   static dispatch with no instance. The
                                       visitor receives detail::type_tag<Payload>
                                       and recovers the payload type via
                                       decltype(tag)::type to call a static method.

DeviceTypeSwitch:
  visit(tag, visitor, fallback)        like visit_type, for stateless operators.
```

Invariant for `visit`/`apply`: `owner.type` always equals the active member's
tag (maintained by construct/copy/assign), so the visitor is always invoked on
the active payload; only a corrupted tag reaches the fallback (or, for `apply`,
does nothing). `visit_type` and `DeviceTypeSwitch::visit` access no payload
storage at all — they map the tag to a type and call a static method.

### Keep payload interfaces uniform

`visit`/`apply` require the visitor to call the **same expression** on every
payload. When a case once differed — a payload with a shorter argument list, or
a case that returned a constant instead of calling the payload — the fix is to
give that payload a small behavior-preserving overload so the whole family shares
one signature, rather than special-casing it in the dispatch. Current examples:

```text
- HardSphereKernel::cross_section(lhs, rhs, relative_speed) ignores the speed and
  forwards to the two-argument form, matching VHS/VSS.
- IsothermalSurfaceInteraction::internal_energy(...) returns the incident energy
  unchanged, matching the Maxwellian signature.
- MaxwellSigmaGenerate::generate(param0, param1) ignores param1, matching
  the other generators.
- Surface/Volume despawn operators take a position-ignoring four-argument
  despawn(...), matching the tracing operator.
```

A new payload should implement the same uniform interface as its siblings; if it
genuinely cannot, prefer adding an overload over reintroducing a `switch`.

### Recipe: add a new case

Stateful union (DeviceVariant):

```text
1. Add the enum case        (e.g. GeometryType)
2. Add the union member     (the *.h operator struct)
3. Add one DeviceVariantCase<Tag::Value, &Owner::member> to the alias
-- the single templated payload constructor and the behavior methods need NO
   change; visit/apply/visit_type pick the case up automatically --
```

The owning operator declares one constrained template constructor
(`explicit Owner(const Payload&)`) that forwards any payload to
`DeviceVariant::construct_payload`, so adding a case does not add a constructor.

Stateless operator (DeviceTypeSwitch):

```text
1. Add the enum case (SpawnType / DespawnType)
2. Add a stateless operator type with the required static method(s),
   matching the uniform signature of its siblings
3. Add the explicit constructor
4. Add one DeviceTypeCase to the *TypeSwitch alias
-- the dispatch method needs NO change --
```

Contrast this with the old shape, where every behavior method carried its own
`switch (type)`: adding one payload meant editing every switch, and a forgotten
arm became a silent wrong result via `default:`. Now a payload that does not
provide the expected method fails to compile.

---

## 7.4 Status

Every operator union in `include/atlas/` dispatches through these primitives;
no behavior method enumerates its cases with a `switch`.

```text
geometry/geometry_operator                     visit
generator/generate_operator                    visit, apply
collider/interaction/surface_interaction_kernel visit
collider/kernel/post_collider_kernel           apply
solver/dsmc/dsmc_kernel                         visit_type, apply
solver/sph/sph_kernel                           visit_type
source/spawn_operator                           DeviceTypeSwitch
sink/despawn_operator                           DeviceTypeSwitch
```

When extending or migrating, keep changes behavior-preserving: a `visit`/`apply`
fallback equals the old `default:` value, and the lifetime/case-list code is left
untouched.
