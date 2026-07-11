# Contributor Guidelines

This directory holds the working guidelines for contributing to Atlas Engine:
how to scope a change, how to write code, and how to build and test. It is the
companion to [`docs/atlas/`](../atlas/), which documents the individual
refactored modules; the guidelines here describe *how to work on it*.

| Document | Scope |
|---|---|
| [workflow.md](workflow.md) | Working agreement: response language, change discipline, when to ask, and git/commit rules. |
| [coding-style.md](coding-style.md) | How to write code: scope, structure, naming, control flow, state, comments, and portability. |
| [build-and-test.md](build-and-test.md) | Build/test policy, CMake presets and options, and test-authoring rules. |
| [dependencies.md](dependencies.md) | External and in-tree dependencies, the benchmark cases, and the C++ examples. |
| [python.md](python.md) | The nanobind `atlas` module: building it, the factory-function API, and how the bindings are organized. |

## Relationship to the Module Docs

The guidelines and the per-module docs are meant to be read together:

- For the structure of an individual module — the tagged-union leaf pattern, its
  leaves, and how to extend it — see [`docs/atlas/`](../atlas/).
- For how to write and ship a change that fits the project, use this directory.

Where a guideline depends on a structural fact (for example the builder pattern),
it links to the relevant module document rather than restating it, so the two
stay consistent.

---

## How the Simulation Framework Fits Together

Atlas is a DSMC (direct simulation Monte Carlo) engine for rarefied gas flow.
This section is the map a newcomer should read first: what the pieces are, and
what happens when you call `System::update()`.

### The three kinds of object

Everything in the engine is one of three things.

**Data.** Two owners hold all simulation state, and nothing else owns any.

- `Fluid` — the per-particle columns: position, velocity, species. Fixed
  capacity; `particle_count()` is the live prefix and `buffer_size()` the
  ceiling. Also carries the material dictionary and the statistical weight.
- `Universe` — the per-cell grid fields: number density, temperature,
  `collision_count`, the DSMC majorant `max_sigma_g`, `max_relative_speed`, and
  `allocated_solver`.

Both store their state objects in a type-keyed `TypeStore`, looked up as
`state<T>()`. See [atlas/fluid](../atlas/fluid/fluid.md) and
[atlas/universe](../atlas/universe/universe.md).

**Policy.** Interchangeable strategy objects the driver calls into: `Source`,
`Generator`, `Collider`, `Sink`, `Codec`, `Solver`, `Observer`. Nearly all of
them are **tagged-union leaf** types — a concrete umbrella wrapping one of
several leaves and dispatching to it, so a `DeviceBuffer` of umbrellas can
dispatch on the device with no virtual call. See [atlas/core](../atlas/core/core.md)
for the pattern itself.

**The driver.** `System` owns one of each and runs the step loop. It is the only
place the ordering below lives.

### The step pipeline

`System::update()` runs six stages in a fixed order, then advances the step
counter and notifies the observer.

```
                     ┌───────────────── System::update() ─────────────────┐
                     │                                                    │
   Source ─┬────────►│  1. emit      spawn particles, fill their state    │
 Generator ─┘        │       │                                            │
                     │       ▼                                            │
  Searcher ─────────►│  2. search    bin particles into cells             │
                     │       │                                            │
                     │       ▼                                            │
     Codec ─────────►│  3. allocate  assign each cell an owning solver    │
                     │       │                                            │
                     │       ▼                                            │
    Solver ─────────►│  4. solve     intra-cell collisions (DSMC/NTC)     │
                     │       │                                            │
                     │       ▼                                            │
  Collider ─────────►│  5. advect    move particles, resolve boundaries   │
                     │       │                                            │
                     │       ▼                                            │
      Sink ─────────►│  6. remove    despawn, compact the buffer          │
                     │       │                                            │
                     └───────┼────────────────────────────────────────────┘
                             ▼
                        Observer.observe(fluid, universe, step)
```

Read down the table for what each stage touches. "Reads" and "writes" name the
state columns, not the objects.

| Stage | Driven by | Reads | Writes |
|---|---|---|---|
| `emit` | `Source[i]` + `Generator[i]` | — | `position`, `velocity`, `species`; grows `particle_count` |
| `search` | `SpatialHashingSearcher` | `position` | `sorted_index`, `cell_start`, `cell_end`; `number_particle` |
| `allocate` | `Codec` | `temperature`, `number_particle` | `allocated_solver` |
| `solve` | `Solver[]` | `velocity`, `species`, cell ranges | `velocity`; `collision_count`, `max_sigma_g`, `max_relative_speed` |
| `advect` | `Collider[]` | `position`, `velocity` | `position`, `velocity`; advances each collider's `Unit` |
| `remove` | `Sink[]` | `position`, `velocity` | shrinks `particle_count`, compacts the columns |

A few orderings are load-bearing rather than incidental:

- **`search` before `solve`.** Collisions are drawn between particles that share
  a cell, so the spatial hash must be current. `solve` captures the searcher's
  view by value into its device kernels.
- **`allocate` before `solve`.** Each solver's loop index is its id; a cell
  belongs to the solver whose id matches its `allocated_solver`. A single solver
  with no partition owns every cell.
- **Particles move before boundaries do.** `advect` traces every particle against
  the colliders' *current* poses, then advances those poses in a second pass. Do
  it the other way and a particle is tested against a wall that has already left.
- **`remove` last.** Compaction reorders the columns, invalidating the sorted
  indices `search` produced; nothing after it may rely on them.

Each stage is a no-op when its object is absent, so a `System` assembled with
only a fluid, a universe and a searcher still steps — it just does nothing
interesting.

### Where the parallelism is

The work runs through `atlas::parallel_for` over an `ATLAS_ALL_DEVICE` lambda —
directly in `advect` and `remove`, and inside the policy objects the other
stages delegate to (`spawn`, `generate`, `classify`, `allocate`, `solve`).
Which backend that lowers to is a build-time switch (`ATLAS_DEVICE_SYSTEM`, see
[build-and-test.md](build-and-test.md)) — Thrust/CUDA on a GPU, TBB on the CPU —
and the engine sources hold no CUDA-only syntax either way.

Two conventions make that possible, and both are worth internalizing before
writing a kernel:

- **Umbrellas are trivially copyable, so a device lambda captures them by value.**
  This is why a kernel leaf must stay a stateless POD and why per-thread mutable
  state (an RNG, say) is passed *by reference* into the call rather than owned by
  the leaf.
- **Owners hand out views.** A `FluidDsmcView`, `UniverseDsmcView` or
  `SpatialHashingSearcherView` is a trivially-copyable bundle of raw device
  pointers. A view aliases its owner's buffers and dies with them — never outlive
  a resize or a rebuild. The two DSMC views expose `is_complete()`; a consumer
  missing a required state warns and does nothing rather than reading a null
  column. The searcher view carries no such method — its consumers null-check
  the pointers directly.

### A minimal assembly

```cpp
auto system = atlas::System::builder()
    .with_fluid(fluid)                 // particle columns
    .with_universe(universe)           // cell grid; the searcher is built from it
    .with_emitter(source, generator)   // a source paired 1:1 with a generator
    .with_collider(collider)           // boundaries
    .with_sink(sink)                   // outflow
    .with_codec(codec)                 // per-cell solver selection
    .with_solver(solver)               // DsmcSolver
    .with_dt(1.0e-6f)
    .build();

for (int step = 0; step < steps; ++step) {
    system.update();
}
```

Start at [atlas/system](../atlas/system/system.md) for the driver and
[atlas/solver](../atlas/solver/solver.md) for the collision physics.
