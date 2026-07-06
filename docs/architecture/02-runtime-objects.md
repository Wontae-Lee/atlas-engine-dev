# 2. Runtime Objects and Ownership

A simulation is assembled from a small set of runtime objects. This document
describes what each one owns, how they relate, and how they are constructed.
These objects live mostly under `atlas::system::`, `atlas::fluid::`,
`atlas::universe::`, and `atlas::geometry::`, with convenience aliases
re-exported under `atlas::`.

---

## 2.1 The Object Graph

```text
System<T>
 ├─ Fluid<T>          particle storage, material/species, generators, states
 ├─ Universe<T>       domain extents, grid resolution, per-cell field states
 ├─ Source<T>         emits particles into inactive capacity
 ├─ Sink<T>           removes particles and compacts survivors
 ├─ Collider<T>       boundary interaction and collision-aware motion
 └─ Orchestrator<T>   coordinates search, codec, measurement, forces, solvers
      ├─ Searcher<T>      neighbor acceleration structure
      ├─ Codec<T>         assigns particles/cells to solvers
      ├─ Measurer<T>      macroscopic field measurement
      └─ Solver<T>[]      DSMC, SPH, or hybrid physics steppers
```

Ownership is expressed with shared pointers. Each runtime type defines a
`*HostPtr<T>` alias (for example `FluidHostPtr<T>`, `UniverseHostPtr<T>`,
`SystemHostPtr<T>`) that resolves to `atlas::host_shared_ptr<T>`. Components
that must be reachable from device kernels hold `device_shared_ptr<T>` and
`DeviceBuffer<T>` members instead.

---

## 2.2 Fluid

`atlas::Fluid<T>` is the particle ensemble and the central data container.

It owns:

```text
- particle storage organized as a dense active prefix (see 2.7)
- per-particle / per-species material properties
- particle generators (emission operators)
- registered fluid states (position, velocity, and custom typed fields)
- an optional observer hook
```

Key contract members:

- `buffer_size()` — total allocated capacity.
- `particle_count()` — active dense prefix length.
- `set_particle_count(n)` — set how many particles are active.
- `state<S>()`, `has_state<S>()`, `emplace_state<S>(...)` — typed state access.

States are held in a type-erased store so that position, velocity, and any
additional per-particle fields can coexist without coupling them into the core
`Fluid` type.

---

## 2.3 Universe

`atlas::Universe<T>` is the Eulerian simulation domain.

It owns:

```text
- domain bounds (lower/upper corner) and cell size
- grid resolution and derived quantities (cell volume, cell count)
- registered universe states (per-cell field data)
- the scene's boundary/region units, one UnitField per consumer role
  (source / sink / collider / measurer)
```

Per-cell field states (for example temperature, bulk velocity, field force,
collision statistics) are produced by the measurer and consumed by solvers.
Like `Fluid`, `Universe` stores these in a typed state store accessed through
`state<S>()` / `emplace_state<S>(...)`.

The `Universe` is also the central owner of the scene's `Unit`s. Each consumer
role has its own `UnitField` (`atlas::UnitField` — a `DeviceBuffer<Unit>` plus a
cached per-unit / scene-level world-space AABB and one-shot pose integration),
registered at build time via `Universe::Builder::with_{source,sink,collider,
measurer}_units(...)` and reachable as `universe.{source,sink,collider,
measurer}_units()`. `Source`/`Sink`/`Collider`/`VolumeMeasurer` do **not** own
their units — they borrow the matching field through the `UniverseHostPtr` they
are built with (see 2.4/2.5). This keeps unit ownership, pose integration, and
world-bound caching in one place instead of duplicated per role.

---

## 2.4 Source and Sink

`atlas::Source<T>` and `atlas::Sink<T>` are the population-changing endpoints.
Their region units (geometry plus optional kinematics) are owned by the
`Universe` (registered via `Universe::Builder::with_source_units(...)` /
`with_sink_units(...)`); each is built with `.with_universe(...)` and borrows the
matching field. They hold only their role-specific operators (spawn/despawn) —
not the units.

- **Source** emits new particles into inactive capacity that lies *behind* the
  active prefix. It never assumes capacity equals population. Emission patterns
  come from the generator operators (uniform, jittered, Maxwellian).
- **Sink** removes particles inside or crossing its regions and compacts the
  survivors back into a dense `[0, particle_count)` prefix, updating
  `particle_count`.

The active-prefix discipline these two depend on is described in 2.7 and in
[06-conventions.md](06-conventions.md).

---

## 2.5 Collider

`atlas::Collider<T>` applies boundary conditions and drives collision-aware
motion. Its boundary units are owned by the `Universe` (registered via
`Universe::Builder::with_collider_units(...)`); the collider is built with
`.with_universe(...)` and borrows that field. It owns only the per-boundary
surface-interaction kernels (for example isothermal or Maxwellian wall models)
and a post-collision motion policy.

When a collider is installed, `System<T>::advect()` delegates motion to it.
When no collider is installed, the system performs plain time integration:
`position += velocity * dt`.

---

## 2.6 Orchestrator and System

`atlas::system::Orchestrator<T>` coordinates the optional per-step services:
neighbor search, codec classification, measurement, field-force/gravity
application, and the solver stages. It holds the `Universe`, `Fluid`,
`Searcher`, `Codec`, `Measurer`, and a list of solvers. Its work is organized
as a fixed pipeline (see [03-simulation-pipeline.md](03-simulation-pipeline.md)).

`atlas::system::System<T>` is the top-level step driver. It owns the runtime
objects above and exposes the step entry point `update()`, which runs the
per-step sequence. Capacity is *not* owned by `System` — particle capacity
belongs to `Fluid`.

---

## 2.7 Active-Prefix Storage Contract

Particle storage separates allocated capacity from the live population:

```text
buffer:        [ 0 .............................. buffer_size )
active prefix: [ 0 ........ particle_count )
inactive tail:            [ particle_count ...... buffer_size )
```

- Only `[0, particle_count)` is active.
- `Source` appends into the inactive tail.
- `Sink` compacts survivors into the active prefix and updates `particle_count`.

When writing setup code or tests: set capacity with
`Fluid<T>::Builder::with_buffer_size(...)`, set `particle_count` explicitly
when active particles should exist, and never assume
`particle_count == buffer_size`.

---

## 2.8 Construction via Builders

Most public types are assembled through a nested `Builder` with fluent
`.with_*()` setters and a `.build()` / `.make_host_shared()` endpoint. For
example, a fluid is built with `Fluid<T>::Builder` using
`.with_buffer_size(...)`, `.with_properties(...)`, `.with_generators(...)`,
and so on. The builder convention is described in
[06-conventions.md](06-conventions.md).
