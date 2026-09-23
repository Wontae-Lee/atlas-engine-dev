# Simulation Pipeline

One `System::update()` performs six phases in a fixed order and then increments
the step counter:

```text
emit -> search -> allocate -> solve -> advect -> remove
```

The core performs no automatic reporting or CSV output.

| Phase | Driven by | Reads | Writes |
|---|---|---|---|
| `emit` | paired `Source` and `Generator` | source state | position, velocity, species; grows `particle_count` |
| `search` | `SpatialHashingSearcher` | position | sorted indices, cell ranges, `number_particle` |
| `allocate` | optional `Codec` | temperature, `number_particle` | `allocated_solver` |
| `solve` | ordered `Solver[]` | velocity, species, cell ranges | velocity and solver-owned cell statistics |
| `advect` | `Collider[]` | position, velocity | position, velocity; advances collider Units |
| `remove` | `Sink[]` | position, velocity | active flags, compacted columns, `particle_count` |

The order is part of the physics contract:

- Search must precede solve because collision kernels consume current cell
  membership.
- Allocate must precede solve because solver indices select which solver owns a
  cell.
- Particle traces observe the colliders' start-of-step poses; collider Units
  advance only after all particle advection completes.
- Removal is last because compaction can reorder every particle column and
  invalidate the searcher's indices.

A missing optional subsystem makes its phase a no-op. Source-spawn and
sink-removal counters describe only the most recently completed phase; consumers
own cumulative statistics and output cadence. See the
[`System` module](../atlas/system/system.md) for phase-level behavior and
persistence.
