# Core C++ modules

Atlas Core lives in `include/atlas/` and `src/atlas/`. Its public objects are
assembled with builders and stepped by [System](system/system.md). The
[architecture overview](../architecture/overview.md) explains ownership, and
the [simulation pipeline](../architecture/simulation-pipeline.md) explains the
order of one update.

| Area | Modules | Purpose |
|---|---|---|
| State and driver | [Fluid](fluid/fluid.md), [Universe](universe/universe.md), [System](system/system.md) | Particle columns, cell fields, and the update pipeline |
| Physical inputs | [Material](material/material.md), [Geometry](geometry/geometry.md), [Unit](unit/unit.md), [Sync](sync/sync.md) | Species, shapes, and posed boundaries |
| Simulation policies | [Source](source/source.md), [Generator](generator/generator.md), [Collider](collider/collider.md), [Sink](sink/sink.md), [Solver](solver/solver.md), [Codec](codec/codec.md), [Searcher](searcher/searcher.md) | Emission, collisions, removal, cell selection, and spatial lookup |
| Storage and execution | [Buffer](buffer/buffer.md), [Container](container/container.md), [Memory](memory/memory.md), [Parallel](parallel/parallel.md), [Scan](scan/scan.md), [Core utilities](core/core.md) | Backend storage, variants, and execution abstractions |
| Math and support | [Math](math/math.md), [Spatial](spatial/spatial.md), [Random](random/random.md), [Sampling](sampling/sampling.md), [Serialization](serialization/serialization.md), [Logging](logging/logging.md) | Numerical operations, persistence, and diagnostics |

The matching headers show exact signatures. Module documents explain
ownership, builder validation, host/device views, and extension points. The
[native C++ example](../../examples/README.md#native-c) shows a complete
construction and update loop.
