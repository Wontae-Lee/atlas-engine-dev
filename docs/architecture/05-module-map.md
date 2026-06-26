# 5. Module Map

This document catalogs the directories under `include/atlas/`. Each entry gives
the module's responsibility and its primary public type(s). For exact
signatures, read the `.h` declaration file in the corresponding directory.

The modules are grouped by the layers introduced in
[01-overview.md](01-overview.md): portability, primitives, spatial, physics,
simulation objects, coordination/driver, and cross-cutting concerns.

---

## 5.1 Portability

### `core/`
Portability layer. `core/macros.h` defines host/device attributes
(`ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE`), inlining and attribute
macros, and `RESTRICT`. `core/detail/device_variant.h` provides the
device-callable variant used to build operator unions. No runtime types.

---

## 5.2 Primitives

### `math/`
Vectors, matrices, and quaternions. `Vector<T,N>` (with `Vector3<T>`,
`Vector4<T>`), `Matrix<T,R,C>` (with `Matrix3x3<T>`, `Matrix4x4<T>`), and
`Quaternion<T>`. Constants and shared element-wise ops live in
`math/constants.h` and `math/detail/`. Convenience aliases are re-exported
under `atlas::`.

### `buffer/`
Backend-agnostic storage aliases: `DeviceBuffer<T>` and `HostBuffer<T>`. See
[04-backend-portability.md](04-backend-portability.md).

### `memory/`
Ownership and raw-pointer access: `host_shared_ptr<T>`, `device_shared_ptr<T>`,
`make_host_shared`, `make_device_shared`, and `raw_pointer_cast`. On CUDA,
`device_shared_ptr` uses managed memory with atomic reference counting.

### `atomic/`
Backend-agnostic atomic operations (CUDA intrinsics on device, `std::atomic_ref`
on the CPU). Used by reference counting and parallel compaction.

### `parallel/`
Execution-policy-parameterized parallel algorithms: `parallel_for`,
`parallel_fill`, `parallel_sort`, with the `ExecutionPolicy` enum.

### `scan/`
Parallel prefix sum: `exclusive_scan`. Used for index compaction.

### `transform/`
Parallel `transform` and `transform_reduce`.

### `remove/`
Parallel conditional removal: `remove_if`. Used by the sink to despawn
particles.

### `sampling/`
Random and deterministic sampling helpers (unit vectors, hemispheres,
Box-Muller normals, hashed sampling) used by generators and collision kernels.

### `shuffle/`
Deterministic permutation via hashing (`ShuffleOperator`), used for reproducible
species selection and sampling.

### `random/`
PRNG abstraction: `default_random_engine<T>`, `uniform_real_distribution<T>`,
and seed constants.

### `container/`
`Container<T,N>` (a CUDA-friendly fixed-size array) and `TypeStore` (the
type-erased store backing fluid/universe states and observer metrics).

### `tuple/`
Backend-agnostic `tuple` (Thrust or `std`), used with zip iteration.

### `iterator/`
`CountingIterator<T>` and `zip_iterator` for composing parallel ranges.

---

## 5.3 Spatial and Geometry

### `geometry/`
Boundary and confinement shapes. `Geometry<T>` is the host-side polymorphic
interface; `GeometryOperator<T>` is the device-callable union. Concrete shapes:
`Box`, `Sphere`, `Cylinder`, `Plane`, `Circle`, `Square`, `Triangle`,
`TriangleMesh`. See `geometry_type.h` for the shape tag.

### `spatial/`
`AxisAlignedBoundingBox<T>`, `Ray<T>`, and the bounding-volume hierarchy under
`spatial/bounding_volume_hierarchy/` (`bvh.h` interface, with `lbvh` and
`sah_bvh` implementations). Provides fast ray/geometry intersection for mesh
boundaries.

### `transform/` (spatial use)
See primitives above; also used to transform geometry and ray data.

### `sync/`
Rigid-body pose handling. `Sync<T>` and `SyncOperator<T>` convert between local
and world coordinates (translation + quaternion rotation) for moving geometry.

### `unit/`
`Unit<T>` bundles a `GeometryOperator<T>` with an optional `SyncOperator<T>` and
optional linear/angular motion. It is the shared region type used by `Source`,
`Sink`, and `Collider`.

### `searcher/`
Neighbor acceleration structures behind the `Searcher<T>` interface:
`SpatialHashingSearcher<T>` (the primary grid-hash searcher), plus
`kdtree_searcher`, `octree_searcher`, and `quadtree_searcher`. Provides sorted
indices, per-cell ranges, and a neighbor stencil.

### `indexer/`
`DevicePairIndexer` — symmetric pair indexing used by DSMC collision pairing.

---

## 5.4 Physics

### `material/`
`MaterialProperties<T>` and `MaterialType` — per-species physical constants
(mass, reference diameter/temperature, viscosity index, energies, DSMC and SPH
parameters). Built with a fluent builder.

### `generator/`
Particle emission strategies. `Generator<T>` (host interface) and
`GenerateOperator<T>` (device-callable union), with `GenerateType` cases:
uniform, jittering, Maxwell-sigma, Maxwell-Boltzmann. `GeneratePayload` carries
emission inputs.

### `measure/`
Macroscopic field measurement. `Measurer<T>` interface with `VolumeMeasurer`
and `BoltzmannMeasurer` implementations, plus `MeasurerProbe`. Populates
`Universe` per-cell field states.

### `codec/`
Solver allocation. `Codec<T>` assigns particles/cells to solver indices;
`KnudsenCodec` switches DSMC/SPH by local Knudsen number, and
`DeepLearningCodec` is the learned-classification variant.

### `collider/`
Boundary interaction and collision-aware motion. `Collider<T>` owns boundary
units and surface-interaction kernels. Subdirectories:
`collider/interaction/` (surface models such as isothermal and Maxwellian),
`collider/kernel/` (post-collision motion policies: fast, precise, dt-remain,
post), and `collider/detail/` (bound cache and collision kernel internals).

### `solver/`
Physics time-stepping. `Solver<T>` is the abstract base owning protected
`_universe`, `_fluid`, and `_searcher`. Subdirectories:
- `solver/dsmc/` — `DsmcSolver<T>` and variants, with the HS/VHS/VSS collision
  kernels and collision statistics.
- `solver/sph/` — `SphSolver<T>` and smoothing kernels (cubic spline, standard,
  Wendland quintic).
- `solver/hybrid/` — `HybridDsmcSphSolver<T>` combining DSMC and SPH regimes.

DSMC solvers cache a `DsmcSolverProbe` in `_probe`; SPH solvers cache an
`SphSolverProbe`. Refresh probe data with `make_probe()` before device launches
when needed.

---

## 5.5 Simulation Objects

### `fluid/`
`Fluid<T>` and `FluidState`. Particle storage, material/species, generators,
and typed states. See [02-runtime-objects.md](02-runtime-objects.md).

### `universe/`
`Universe<T>` and `UniverseState`. Domain extents, grid resolution, and per-cell
field states.

### `source/`
`Source<T>` — particle injection into the inactive tail. `source/detail/` holds
the emitter, cache builder, species shuffler, and probe builder.

### `sink/`
`Sink<T>` — particle removal and active-prefix compaction. `sink/detail/` holds
the despawner, compactor, unit-bounds cache, and probe builder; the
`despawn_operator` variants implement volume/surface/tracing removal.

---

## 5.6 Coordination and Driver

### `orchestrator/`
`Orchestrator<T>` — coordinates the per-step pipeline (search, classification,
measurement, force, solve). `orchestrator/detail/` holds the pipeline, the
force applier, and the probe builder. See
[03-simulation-pipeline.md](03-simulation-pipeline.md).

### `system/`
`System<T>` — top-level step driver. Owns the runtime objects and exposes
`update()`.

---

## 5.7 Cross-Cutting

### `observer/`
Optional diagnostics. `Observer` holds typed sensor metrics (`SensorMetrics`,
with `SourceSensorMetrics` / `SinkSensorMetrics`) and can export them (for
example to CSV).

### `logging/`
Optional `Logger` with severity levels; compiled implementation in
`src/logging/`. Enabled by build configuration.

### `serialization/`
Binary checkpoint/restore. `FluidBinarySnapshot<T>` and
`UniverseBinarySnapshot<T>` plus save/load functions; the protobuf-backed
implementation is compiled in `src/serialization/`.

---

## 5.8 Vizkit (Optional Visualization)

Vizkit is the OpenGL visualization layer. It is *not* part of the header-only
core under `include/atlas/`; it lives in `src/vizkit/` and is compiled only when
`ATLAS_ENABLE_VIZKIT` is defined.

- `vizkit::Viewer<T>` stores a `SystemHostPtr<T>` and reads `system()->dt()`.
- `Viewer<T>::Builder` uses `.with_system(...)`, `.with_size(...)`,
  `.with_title(...)`, and `.with_fullscreen(...)`.
- Layer updates receive `(GLFWwindow*, Camera&, T dt)`.

---

## 5.9 Umbrella Header

`include/atlas/atlas.h` aggregates every public header. It is generated by
`tools/generate_headers.py`, which walks `include/atlas/`, collects every `.h`
(and `.cuh`) file except the project umbrella and `math.h`, classifies Vizkit
headers separately, and writes the sorted `#include` list. Do not hand-edit the
umbrella header; regenerate it with the tool when adding or moving public
headers.
