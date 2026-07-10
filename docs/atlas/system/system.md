# System

`System` is the top of the engine: the simulation driver that owns every other
subsystem and runs the world one timestep at a time. It aggregates the particle
[`Fluid`](../fluid/fluid.md), the spatial [`Universe`](../universe/universe.md)
grid and its per-cell state, the spatial-hashing searcher, the emission
[`Source`](../source/source.md) / generator pairs, the physics
[`Solver`](../solver/solver.md) list, the [`Collider`](../collider/collider.md)
and [`Sink`](../sink/sink.md) boundaries, the per-cell
[`Codec`](../codec/codec.md), and the diagnostic [`Observer`](../observer/observer.md).
A single call to `System::update()` advances the whole simulation by one fixed
timestep.

Unlike most documented modules, `System` is not a tagged-union leaf type. It is a
plain move-only orchestrator: every subsystem is held by a host smart pointer or
a device buffer the system owns outright, and each pipeline phase launches its
own device kernels internally. Construct one through `System::Builder`.

## Files

| File | Role |
|---|---|
| `include/atlas/system/system.h` | `System` driver, its pipeline-phase methods, and the nested `System::Builder`; `SystemHostPtr` alias |
| `src/atlas/system/system.cu` | Out-of-line definitions: constructor, state provisioning, the six pipeline phases, `save`, and `Builder` |

## `System::Builder`

The builder accumulates subsystems through chainable `with_*` setters, then
`build()` validates and constructs the `System` by value (or `make_host_unique()`
wraps it in a `SystemHostPtr`).

| Setter | Effect | Cardinality |
|---|---|---|
| `with_fluid(FluidHostPtr)` | Particle store | required, one |
| `with_universe(UniverseHostPtr)` | Spatial grid + per-cell state | required, one |
| `with_solver(SolverHostPtr)` | Appends a physics solver | zero or more, insertion order |
| `with_emitter(SourceHostPtr, GeneratorHostPtr)` | Appends a source paired with its generator | zero or more, index-aligned |
| `with_collider(const Collider&)` | Appends a collision boundary (copied into a host buffer) | zero or more |
| `with_sink(const Sink&)` | Appends a removal boundary (copied into a host buffer) | zero or more |
| `with_codec(CodecHostPtr)` | Per-cell solver selector | optional, may be null |
| `with_observer(ObserverHostPtr)` | Diagnostic sampler | optional, may be null |
| `with_dt(float)` | Fixed timestep in seconds | defaults to `0.01f` |

`with_emitter` pushes onto the source and generator lists together, so source `i`
always keeps its generator `i`; `emit()` relies on that alignment.

### Validation rules

`build()` calls `validate()` first, which throws `std::runtime_error` on the
first failed requirement:

- a fluid must be set (non-null);
- a universe must be set (non-null);
- `dt` must be strictly positive (`_dt > 0.0f`);
- the source and generator counts must be equal, and no source or generator
  entry may be null;
- no solver entry may be null.

Colliders, sinks, the codec, and the observer are all optional, and a system
with zero sources runs fine (it just never emits).

### What the constructor derives

Once `validate()` passes, the `System` constructor takes ownership of the host
pointers and copies the collider/sink host buffers into device buffers, then
derives dependent state:

- if a universe is present, it builds the spatial-hashing searcher over it (the
  searcher indexes the grid, so it cannot exist before the universe does);
- if an observer is present, it sizes the observer's counter matrices via
  `resize_counters(source_count, sink_count, species_count)`, where
  `species_count` comes from the fluid's material dictionary
  (`_fluid->materials()->size()`) or falls back to `1` when there is no
  dictionary — a single implicit species, so the counter layout is never empty;
- it calls `initialize_states()` to provision the universe state columns the
  configured solvers need.

## The step pipeline

`System::update()` runs six phases in a fixed order, increments the step
counter, and then lets the observer sample the result:

```cpp
void System::update() {
    emit();       // 1. spawn new particles from every source
    search();     // 2. bin live particles into universe cells
    allocate();   // 3. choose a solver per cell
    solve();      // 4. run each physics solver over the fluid
    advect();     // 5. move particles, resolve collisions, advance colliders
    remove();     // 6. despawn particles at sinks and compact the fluid
    ++_step;
    if (_observer) {
        _observer->observe(*_fluid, *_universe, _step);
    }
}
```

The observer runs *after* the step counter is incremented, so it samples with the
new (one-based) step index; its own interval gate decides whether that step
actually writes anything (see [`observer.md`](../observer/observer.md)). Each
phase below is also callable on its own, but the ordering in `update()` is the
contract.

### 1. `emit`

Spawns new particles from every source into the fixed-capacity fluid buffer.
It is a no-op without a `FluidPositionState`. New particles are appended at the
current live count and grow it in place:

- iterate the sources in order; before each source, if the live count has already
  reached `buffer_size`, stop — once the buffer is full, further spawns are
  dropped;
- `source->spawn(positions, count)` writes new positions starting at the current
  live count and returns how many it wrote; a non-positive return skips this
  source;
- the paired generator fills velocity and species for exactly those spawned
  slots (`generate(velocities, species, count, spawned)`); its own return is
  discarded because `spawned` already fixes the slot range;
- `record_spawned(i, count, spawned)` reports the spawn to the observer against
  the range starting at the pre-increment offset;
- the live count advances by `spawned`.

After all sources have emitted, `set_particle_count(count)` publishes the new
count, and every source boundary is advanced by `_dt` so the next step sees the
moved emitters.

### 2. `search`

A no-op without a searcher. It delegates to the spatial-hashing searcher, which
classifies each live particle into its universe cell and publishes the per-cell
particle counts into the `UniverseNumberParticleState` column
(`searcher->classify(positions, number_particle, particle_count)`).

### 3. `allocate`

A no-op without a codec. The codec reads each cell's temperature
(`UniverseTemperatureState`) and number density (`UniverseNumberParticleState`)
and writes the owning solver index into `UniverseAllocatedSolverState`, which
`solve()` later honours.

### 4. `solve`

A no-op unless a fluid, a universe, a searcher, and at least one solver are all
present. It gathers the trivially-copyable searcher view once
(`_searcher->view()`) so device kernels can capture it by value, then runs every
solver in insertion order. Each solver receives its own loop index as its id:

```cpp
for (std::size_t i = 0; i < _solvers.size(); ++i) {
    _solvers[i]->solve(*_fluid, *_universe, searcher_view, static_cast<int>(i), _dt);
}
```

A cell compares that id against its allocated-solver selection (written by
`allocate`) to decide whose work applies to it, which is how the codec's per-cell
choice is honoured.

### 5. `advect`

Moves particles and resolves collisions. A no-op unless both position and
velocity states exist. It runs in two passes.

The first pass is one kernel per particle:

- compute `sweep_length = velocity.length() * dt`. A particle whose sweep is
  shorter than `atlas::eps` cannot cross a boundary this step, so it skips the
  collision search entirely and leaves its position untouched;
- **broad phase** — build the swept segment (`Ray sweep(position, velocity)`)
  and test it against each collider's world AABB (`collider.bound()`). An
  unbounded geometry such as a plane has no valid AABB (`bound.is_valid()` is
  false), so it *skips* the broad phase rather than rejecting everything;
  a bounded collider is culled when the swept segment misses its box or enters it
  past `sweep_length`;
- **narrow phase** — trace the survivors (`collider.trace(position, velocity, dt)`)
  and keep the nearest hit by `distance`;
- **resolve** — if a nearest hit exists, `collider.collide(hit, position,
  velocity, dt)` updates position and velocity against it; otherwise integrate
  straight ahead (`position += velocity * dt`).

The second pass advances every moving collider by `_dt`, in a separate kernel run
*after* all particles have been traced. The order is load-bearing: every particle
in this step is tested against the colliders at their start-of-step positions, so
the advance must not happen until tracing is finished — otherwise particles
processed later would collide against boundaries that had already moved forward,
making the result depend on particle ordering.

### 6. `remove`

Despawns particles at sinks and compacts the fluid. Runs the removal path only
when there are sinks, live particles, and both position and velocity states:

- `mark_survivors(particle_count)` flags each live particle `1` (survives) or `0`
  (despawned) against the sinks — see below;
- `_fluid->compact()` keeps exactly the survivors, in their original relative
  order, and updates the live count (the fluid owns the scan, the gather indices,
  and every state column it gathers).

Independently, whenever any sinks exist, every sink boundary is advanced by
`_dt` — even on a step that removed nothing, because a sink may still be moving.

## `mark_survivors` and `record_spawned` — why they are public

Both `mark_survivors` and `record_spawned` are internal steps of `remove` and
`emit`, yet they are declared `public`. This is deliberate and non-obvious. Each
body launches an extended `__host__ __device__` lambda, and **nvcc forbids such a
lambda inside a private or protected member function**. The CI job on the
`tbb-nvcc-debug` preset exists precisely to catch this — a TBB-only build (host
compiler) would accept the private form and hide the error. See
[build-and-test.md §2](../../guidelines/build-and-test.md). They are not meant to
be called directly outside the pipeline; the tests exercise them only to inspect
the survivor flags in isolation.

- `mark_survivors(particle_count)` — for each particle, tests each sink's despawn
  predicate; the first sink to claim it clears the active flag and, when an
  observer with a correctly shaped despawn counter is present, atomically bumps
  the per-sink, per-species tally. Particles that survive every sink are flagged
  `1`. The flags become the scan input `compact()` consumes.
- `record_spawned(source_index, offset, count)` — a no-op without an observer,
  with a zero count, or when the species state or counter layout does not match.
  Otherwise it atomically increments the `(source, species)` counter for each
  newly spawned slot, after a bounds check on the species id.

## State initialization

`initialize_states()` provisions the universe state columns the configured
solvers need. It is a no-op without a universe. It always provisions the
`UniverseAllocatedSolverState` column (every configuration needs the per-cell
solver-selection column, regardless of solver kind), then walks the solver list
and, for each DSMC solver, calls `initialize_dsmc_states()`, which ensures the
`UniverseNumberParticleState`, `UniverseMaxRelativeSpeedState`,
`UniverseMaxSigmaGState`, and `UniverseCollisionCountState` columns exist and are
sized to the cell count. It is idempotent: a column that already exists is
resized in place if the cell count changed rather than reallocated blindly, so it
is safe to call again after the grid is resized. The constructor calls it once.

## Snapshots — `save()`

`save(directory)` writes a **binary** snapshot of the current step for offline
inspection, distinct from the observer's CSV output. It creates a per-step
subdirectory named by `snapshot_directory_name(step)` — the string
`"time_step_<step>"` — under `directory`, then serializes `fluid.bin` and
`universe.bin` into it via the protobuf snapshot serializers. It throws
`std::runtime_error` on an I/O failure. This is the durable protobuf format; the
[`Observer`](../observer/observer.md) is the separate, periodic CSV diagnostics
path.

## Assembling a simulation

The minimal loop is: build the fluid and universe, attach whatever sources,
solvers, colliders, sinks, a codec, and an observer the scenario needs, then call
`update()` repeatedly.

```cpp
System system = System::builder()
    .with_fluid(std::move(fluid))            // required
    .with_universe(std::move(universe))      // required
    .with_emitter(std::move(source), std::move(generator))  // paired; repeatable
    .with_solver(std::move(solver))          // repeatable, runs in order
    .with_collider(collider)                 // repeatable
    .with_sink(sink)                         // repeatable
    .with_codec(std::move(codec))            // optional
    .with_observer(std::move(observer))      // optional
    .with_dt(1.0e-3f)                        // must be positive
    .build();                                // validates, then constructs

for (std::size_t s = 0; s < steps; ++s) {
    system.update();                         // emit → search → allocate → solve → advect → remove
}
```

The builder wires the searcher and provisions the universe state columns for you,
so no manual state registration is needed before the loop. Observers write CSV on
their own interval; call `system.save(dir)` where a binary snapshot is wanted.
