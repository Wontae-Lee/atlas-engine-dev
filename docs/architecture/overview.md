# Atlas Architecture

Atlas Core is a self-contained C++ computational engine under `include/atlas/`
and `src/atlas/`. It owns physical state and advances that state; it has no
Python, Interactive, rendering, JSON, or application-output dependency.

## Roles

**Data owners** hold the canonical simulation state:

- [`Fluid`](../atlas/fluid/fluid.md) owns fixed-capacity per-particle columns.
  `particle_count()` selects the live prefix.
- [`Universe`](../atlas/universe/universe.md) owns the spatial grid and per-cell
  fields.

Both use type-keyed state stores and expose typed state access.

**Policies** implement interchangeable behavior: `Source`, `Generator`,
`Collider`, `Sink`, `Codec`, and `Solver`. Most use the tagged-union leaf pattern
described in [`Core`](../atlas/core/core.md), allowing compile-time-selected TBB
or CUDA execution without virtual device dispatch.

**The driver**, [`System`](../atlas/system/system.md), owns Fluid, Universe, the
searcher, policies, and the fixed timestep. It is the only owner of the ordered
step pipeline.

## Construction and validation

Core builders own Atlas semantic and numerical validation. `Fluid::Builder`
stages the initial live particle columns and checks their lengths, capacity,
weight, and material indices. `Universe::Builder` derives the grid, checks its
dimensions and cell count, and installs initial cell columns. Policy builders
validate their own parameters where those builders define rules. Material leaves
are direct value types; their dictionary builder checks that a dictionary is
nonempty, but does not validate every physical property. `System::Builder`
checks cross-object assembly:
required Fluid and Universe, positive timestep, non-null solvers, and paired
sources and generators. A successful build transfers the assembled objects to
`System`; it also provisions solver-owned cell states.

The Interactive [JSON decoder](../frontends/interactive.md) checks field shape,
types, and required fields. Its `CoreFactory` (also aliased as `SystemFactory`)
maps typed configuration to these same Core builders. It does not maintain a
second numerical rule set. Builder exceptions become failed Interactive
responses, so `validate` and `create` share construction semantics.

## Ownership boundaries

Builders transfer ownership into System. Host smart pointers keep policies and
shared resources alive; backend buffers own simulation data. Device views and
raw pointers alias those owners and must not outlive a resize, rebuild, or owner.
Consumers read state through System's Fluid and Universe accessors and decide
when to analyze, serialize, report, or render it.

For example, a `TriangleMesh` host owner keeps its BVH and triangle buffers
alive while a geometry view refers to them. The CPU backend uses host buffers;
the CUDA backend uses device buffers. The same ownership rule applies to both.

Python and Interactive are sibling consumers, described in
[frontend boundaries](frontends.md). Detailed module structure remains under
[`docs/atlas/`](../atlas/).
