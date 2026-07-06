# 3. Simulation Pipeline

This document follows a single simulation step from the top-level driver down
through the orchestrator pipeline. The control flow described here is the
contract that solver, source, sink, and collider changes must preserve.

---

## 3.1 The Step Entry Point

A step is driven by `System<T>::update()`:

```text
System<T>::update()
  emit();          // source stage
  orchestrate();   // orchestrator stage
  advect();        // motion stage (collider or plain integration)
  remove();        // sink stage
```

Each sub-step is guarded: it runs only when the corresponding component is
installed.

```text
emit()        -> if (source)       source->update(dt)
orchestrate() -> if (orchestrator) orchestrator->update(dt)
advect()      -> if (collider)     collider->update(dt)
                 else              time_integration()
remove()      -> if (sink)         sink->update(dt)
```

`time_integration()` is the no-collider fallback motion. It advances active
particles only and is a no-op when there is no fluid, no positive `dt`, no
cached position/velocity state, or zero active particles:

```cpp
position += velocity * dt;   // over [0, particle_count)
```

---

## 3.2 The Orchestrator Pipeline

`orchestrator->update(dt)` runs a fixed pipeline of stages, in this order:

```text
Orchestrator pipeline (per step)
  1. search          rebuild the neighbor acceleration structure
  2. classification  run the codec to assign particles/cells to solvers
  3. measurement     compute macroscopic per-cell fields (measurer)
  4. force           apply gravity and field forces
  5. solve           run the installed solvers
```

Stage notes:

- **search** — if a searcher is installed, it is invalidated and rebuilt so
  that downstream neighbor queries (DSMC pairing, SPH summation) see current
  particle positions.
- **classification** — the codec maps particles or cells to solver indices
  (for example DSMC vs SPH by local Knudsen number). Solvers then operate only
  on their assigned allocation.
- **measurement** — the measurer aggregates per-cell macroscopic quantities
  used by force application and by solvers.
- **force** — gravity and field forces are applied to particle velocities for
  the cells that carry those fields. Cells without an applicable force are
  skipped.
- **solve** — each installed solver advances the particles in its allocation.

Each stage is a no-op when its component is absent, so a minimal configuration
(for example a pure ballistic test) can omit the searcher, codec, measurer, and
solvers entirely.

---

## 3.3 End-to-End Data Flow

The diagram below ties the stages to the objects they read and write.

```text
emit
  Source<T>::update(dt)
    └─ append new particles into the fluid inactive tail
       (positions / velocities / species from generator operators)

orchestrate
  Orchestrator<T>::update(dt)
    ├─ search       Searcher<T>: sort particles by spatial key,
    │                            build cell ranges + neighbor stencil
    ├─ classify     Codec<T>:    write per-particle/per-cell solver allocation
    ├─ measure      Measurer<T>: write per-cell Universe field states
    ├─ force        apply gravity + field force to velocities
    └─ solve        Solver<T>[]: read neighbors + fields, update particle state
                                 (DSMC collisions, SPH forces, or hybrid)

advect
  Collider<T>::update(dt)   boundary interaction + collision-aware motion
    └─ or, without a collider: position += velocity * dt

remove
  Sink<T>::update(dt)
    └─ mark particles in/leaving sink regions, then compact survivors
       into [0, particle_count) and update particle_count
```

---

## 3.4 Invariants the Pipeline Must Preserve

Changes to any stage should keep these invariants intact:

```text
- The active prefix [0, particle_count) stays dense after remove().
- Source only writes into the inactive tail; it does not move active
  particles.
- Allocation filtering only restricts which cells a solver touches. It must
  not clear collision statistics or universe state owned by another solver.
- A guard branch (missing component, empty population) updates only the state
  owned by its own step; it does not silently repair unrelated state.
- Motion is owned by exactly one path per step: the collider when installed,
  otherwise time_integration().
```

See [06-conventions.md](06-conventions.md) for the broader coding contract and
the solver-specific guidelines.
