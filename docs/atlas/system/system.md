# System

`System` is the simulation driver. It owns the particle [`Fluid`](../fluid/fluid.md),
the spatial [`Universe`](../universe/universe.md), the spatial-hashing searcher,
source/generator pairs, solvers, colliders, sinks, and the optional per-cell
[`Codec`](../codec/codec.md). One call to `System::update()` advances the physical
simulation by one fixed timestep.

`System` is a plain move-only orchestrator rather than a tagged-union leaf. Its
subsystems are held by host smart pointers or device buffers, and its pipeline
phases launch backend-selected kernels internally.

## Files

| File | Role |
|---|---|
| `include/atlas/system/system.h` | Driver API, pipeline phases, accessors, and `System::Builder` |
| `src/atlas/system/system.cu` | Construction, state provisioning, the six pipeline phases, snapshots, and builder definitions |

## Builder

| Setter | Effect | Cardinality |
|---|---|---|
| `with_fluid(FluidHostPtr)` | Particle store | required, one |
| `with_universe(UniverseHostPtr)` | Spatial grid and per-cell state | required, one |
| `with_solver(SolverHostPtr)` | Appends a physics solver | zero or more |
| `with_emitter(SourceHostPtr, GeneratorHostPtr)` | Appends an aligned source/generator pair | zero or more |
| `with_collider(const Collider&)` | Appends a collision boundary | zero or more |
| `with_sink(const Sink&)` | Appends a removal boundary | zero or more |
| `with_codec(CodecHostPtr)` | Sets the per-cell solver selector | optional |
| `with_dt(float)` | Sets the fixed timestep in seconds | defaults to `0.01f` |

`build()` rejects a missing fluid or universe, a non-positive timestep, null
solvers, and missing or null entries in source/generator pairs. Construction
builds the searcher over the universe and provisions the universe state columns
required by the configured solvers.

`Builder::validate()` checks this assembly without advancing or saving state.
`build()` moves the Fluid, Universe, and source/generator pointers into the
new System; a builder whose required pointers have moved needs reconfiguration
before another build. Policy arrays retain their declared order. Core owns
these checks for both native callers and Interactive: an Interactive `Session`
owns a System through `CoreFactory`, but System has no Session dependency.

## Step pipeline

```cpp
void System::update() {
    emit();
    search();
    allocate();
    solve();
    advect();
    remove();
    ++_step;
}
```

`update()` performs no filesystem output and applies no reporting cadence. A
consumer can read `fluid()` and `universe()` after any step and decide how to
analyze or export that state.

### `emit`

Each source appends positions at the current live-particle count until the fixed
fluid capacity is full. Its paired generator fills velocity and species for the
same slots. The phase publishes the new particle count and advances every source
boundary by `dt`. `source_spawned_last_step()` exposes one raw count per source
for the most recently completed `emit()` phase. The counts reset at the start of
the next `emit()` call and do not apply any reporting policy.

### `search`

The spatial-hashing searcher bins live particle positions into universe cells
and writes the per-cell particle count. The phase is a no-op without a searcher.

### `allocate`

The optional codec reads the cell state and selects a solver index for every
cell. The phase is a no-op without a codec.

### `solve`

Every solver runs in insertion order with its list index as the solver id. The
solver compares that id with the cell's allocated-solver state. The phase needs
a fluid, universe, searcher, and at least one solver.

### `advect`

For each particle, `advect()` traces the timestep sweep against collider bounds,
resolves the nearest surface hit, or integrates `position += velocity * dt` when
there is no hit. Collider motion is advanced in a separate pass after all
particles have been traced so every particle observes the same start-of-step
boundary state.

### `remove`

When sinks and live position/velocity states are present, `mark_survivors()`
sets each leading active flag to `0` when the first sink despawns the particle
and to `1` otherwise. `Fluid::compact()` then preserves the survivors in their
original order and updates `particle_count`. Sink boundaries advance even when
no particle is removed.

`mark_survivors()` is public because its body launches an extended
`__host__ __device__` lambda, which nvcc rejects inside a private or protected
member function. It remains an internal pipeline step in normal use.

`sink_removed_last_step()` copies one raw count per sink from the most recently
completed `remove()` phase. A particle claimed by overlapping sinks is counted
only for the first sink that removes it, matching survivor selection. The
counters reset at the start of the next `remove()` call. They are simulation
events only; cumulative statistics, output intervals, and files remain consumer
responsibilities.

## State access and persistence

`fluid()` and `universe()` return the objects owned by the system. These are the
canonical access paths for particle and cell state. `save(directory)` writes the
current fluid and universe protobuf snapshots below `time_step_<step>/`; this is
explicit core state persistence requested by the caller, not periodic reporting.
The last-phase source and sink counter accessors allow consumers to build
statistics without making `System::update()` perform reporting or I/O.
The read-only `source_syncs()`, `collider_syncs()`, and `sink_syncs()` snapshots
expose current boundary poses to native consumers. CUDA configurations copy only
the small collider/sink metadata arrays to the host; particle state remains on
the selected backend.

## Assembly

```cpp
System system = System::builder()
    .with_fluid(std::move(fluid))
    .with_universe(std::move(universe))
    .with_emitter(std::move(source), std::move(generator))
    .with_solver(std::move(solver))
    .with_collider(collider)
    .with_sink(sink)
    .with_codec(std::move(codec))
    .with_dt(1.0e-3f)
    .build();

for (std::size_t step = 0; step < steps; ++step) {
    system.update();
}
```
