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

## Ownership boundaries

Builders transfer ownership into System. Host smart pointers keep policies and
shared resources alive; backend buffers own simulation data. Device views and
raw pointers alias those owners and must not outlive a resize, rebuild, or owner.
Consumers read state through System's Fluid and Universe accessors and decide
when to analyze, serialize, report, or render it.

Python and Interactive are sibling consumers, described in
[frontend boundaries](frontends.md). Detailed module structure remains under
[`docs/atlas/`](../atlas/).
