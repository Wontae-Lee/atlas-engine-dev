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
macros, and `RESTRICT`. `core/device_variant.h` provides the
device-callable variant used to build operator unions. No runtime types.

---

## 5.2 Primitives

### `math/`
Float vectors, matrices, and quaternions as plain non-template classes:
`Float3` (float), `Int3` (int grid coordinates), `Float3x3`,
`Quaternion`, plus `Bool3` (element-wise comparison result consumed by
`all`/`any`/`none`). Float constants and scalar helpers live in
`math/constants.h`. The entire module is device-hot and therefore
header-inline (definitions in the headers, no `.cu` translation units); this
is the documented exception to the `.h`/`.cu` split.

### `buffer/`
Backend-agnostic storage aliases: `DeviceBuffer<T>` =
`thrust::device_vector<T>` and `HostBuffer<T>` = `thrust::host_vector<T>`. The
active Thrust device system decides where `DeviceBuffer` memory lives. See
[04-backend-portability.md](04-backend-portability.md).

### `memory/`
Ownership, raw-pointer access, and host/device copies: `host_shared_ptr<T>`,
`device_shared_ptr<T>`, `make_host_shared`, `make_device_shared`,
`device_ptr<T>` (= `thrust::device_ptr<T>`), `raw_pointer_cast`, and the
`copy_host_to_device` / `copy_device_to_host` helpers (implemented with
`thrust::copy_n`, so they work under both device systems). On CUDA,
`device_shared_ptr` uses managed memory with atomic reference counting.

### `parallel/`
Execution-policy-parameterized parallel algorithms: `parallel_for`,
`parallel_fill`, `parallel_sort` / `parallel_sort_by_key`, with the
`ExecutionPolicy` enum (`serial`, `host`, `device`). Each dispatches at compile
time to `thrust::seq` / `thrust::host` / `thrust::device`.

### `scan/`
Parallel prefix sum: `exclusive_scan`. Used for index compaction.

### `transform/`
Parallel `transform` and `transform_reduce`.

### `remove/`
Parallel conditional removal: `remove_if<ExecutionPolicy>`. Used by the sink to
despawn particles.

### `sampling/`
Random and deterministic sampling helpers (unit vectors, hemispheres,
Box-Muller normals, hashed sampling) used by generators and collision kernels.

### `shuffle/`
Deterministic permutation via hashing (`Shuffle`), used for reproducible
species selection and sampling.

### `random/`
PRNG abstraction over Thrust: `default_random_engine` (non-template alias),
`uniform_real_distribution<T = float>`, and float/integer seed constants in
`seed.h`.

### `container/`
`Container<T,N>` (a CUDA-friendly fixed-size array, with `Container2/3/4`
aliases and `TriangleContainer4` = `Container<Float3, 4>`) and `TypeStore`
(the type-erased store backing fluid/universe states and observer metrics).

### `tuple/`
`atlas::tuple` = `thrust::tuple`, with `make_tuple` / `get` wrappers, used with
zip iteration.

### `iterator/`
`counting_iterator<T>` and `zip_iterator` (Thrust aliases) for composing
parallel ranges.

---

## 5.3 Spatial and Geometry

### `geometry/`
Boundary and confinement shapes. Each concrete shape (`Box`, `Sphere`,
`Cylinder`, `Plane`, `Circle`, `Square`, `Triangle`) is a device-callable VALUE
type: it embeds its parameters by value and carries its own `ATLAS_ALL_DEVICE`
query methods (`closest_point`, `signed_distance`, `is_inside`, `bound`,
`trace`, ...) — there is no separate `*GeometryOperator` struct and no abstract
`Geometry` base (CUDA has no device virtual dispatch, so shapes are plain
values). `GeometryOperator` (see `geometry_type.h` for the tag) is the
device-callable tagged union that holds one shape value and dispatches queries
to it; construct it from a shape value, `GeometryOperator(box)`. `TriangleMesh`
is the exception: it OWNS its vertex/index/BVH buffers, so it stays a host class
that produces a lightweight `TriangleMeshGeometryOperator` pointer-view (via
`mesh.make_device_geometry_view()`), which is the union's `TriangleMesh` case.
Host-only builders live in `src/atlas/geometry/*.cu`.

### `spatial/`
`AxisAlignedBoundingBox` (alias `AABB`), `Ray`/`HitSurface`, and the
bounding-volume hierarchy under `spatial/bounding_volume_hierarchy/` (`bvh.h`
interface `BVH`, with `LBVH` and `SAHBVH` implementations;
build code in `src/atlas/spatial/bounding_volume_hierarchy/*.cu`). Provides
fast ray/geometry intersection for mesh boundaries.

### `transform/` (spatial use)
See primitives above; also used to transform geometry and ray data.

### `sync/`
Rigid-body pose handling. `Sync` is a device-callable VALUE type (header-inline)
that converts between local and world coordinates (translation + quaternion
rotation) for moving geometry — it absorbed the former `SyncOperator`, so there
is no separate operator type. Its host-only builder lives in
`src/atlas/sync/sync.cu`.

### `unit/`
`Unit` bundles a `Geometry` with a `Sync` and optional linear/angular motion
(accessors `geometry()` / `sync()`), and exposes `world_bound()` — its
geometry AABB mapped into world space through its sync. It is the shared region
type used by `Source`, `Sink`, `Collider`, and `VolumeMeasurer`. `Unit` is a
device-copyable value type (header-inline); the host-only builder lives in
`src/atlas/unit/unit.cu`. `UnitField` (`unit_field.h` / `unit_field.cu`) is a
set of `Unit`s owned as one `DeviceBuffer<Unit>` together with the shared
per-step machinery every consumer needs — pose integration (`advance`) and a
cached per-unit / scene-level world-space AABB (`refresh_bounds`,
`unit_bounds`/`scene_bound`/`covers_units`). The `Universe` owns one `UnitField`
per consumer role; the roles borrow it (see `universe/`).

### `searcher/`
Neighbor acceleration structures behind the host-only `Searcher` interface:
`SpatialHashingSearcher` (the default grid-hash searcher), plus
`KdTreeSearcher`, `OctreeSearcher`, and `QuadtreeSearcher`. The base
`Searcher` owns the neighbor buffers and exposes the query surface (`build`,
`indices`, `cell_start`/`cell_end`, `neighbor_*`) as plain-host virtuals;
the static grid helpers (`cell_for`, `linear_key`, `contains_cell`,
`search_radius_for`) use `Int3` cells and stay device-callable
header-inline together with the neighbor count/write functors. Concrete
searchers override `build`; host-side build/builder logic lives in
`src/atlas/searcher/*.cu`. The consumers that need neighbors —
`Orchestrator`, `Codec`, and `Measurer` — depend on `SearcherHostPtr` (the
base), so the user selects the neighbor-search strategy simply by building
and passing the searcher of their choice to `with_searcher(...)`.

### `indexer/`
`DevicePairIndexer` — symmetric pair indexing used by DSMC collision pairing.

---

## 5.4 Physics

### `material/`
`MaterialProperties` and `MaterialType` — per-species physical constants
(mass, reference diameter/temperature, viscosity index, energies, DSMC and SPH
parameters). Built with a fluent builder. This module is the reference example
of the `.h`/`.cu` split: the header declares the record and the builder;
`src/atlas/material/material_properties.cu` defines the host-only builder
logic.

### `generator/`
Particle emission strategies. `Generator` (host-only interface) and
`Generate` (device-callable union, header-inline), with `GenerateType`
cases: uniform, jittering, Maxwell-sigma, Maxwell-Boltzmann. The payload
operators (`generate_payload.h`) own their parameters by value and are
header-inline. Concrete host generators (`UniformGenerator`,
`JitteringGenerator`, `MaxwellSigmaGenerator`, `MaxwellBoltzmannGenerator`)
declare in headers and define their builders/methods in
`src/atlas/generator/*.cu`.

### `measure/`
Macroscopic field measurement. `Measurer` is a host-only virtual interface
(`measure`, `measure(dt)`, `measure_mode`) owning protected `_universe`,
`_fluid`, `_searcher`, and a `MeasurerProbe` — a plain float value struct of
pointer views into the universe/fluid state buffers and searcher grid arrays,
rebuilt by `make_probe`. Implementations: `BoltzmannMeasurer` (per-cell bulk
velocity, thermal energy, number density, temperature) and `VolumeMeasurer`
(unit-occupancy cell volume with `Int3` cell regions). Populates
`Universe` per-cell field states. Host definitions and the measurement
kernels (device lambdas confined to non-virtual helpers) live in
`src/atlas/measure/*.cu`.

### `codec/`
Solver allocation. `Codec` (host-only virtual base, float) assigns cells to
solver indices and caches a `CodecProbe` — a header-inline value struct whose
pointers are views into universe/fluid/searcher-owned `DeviceBuffer`s,
rebuilt by `make_probe()`. `KnudsenCodec`
switches DSMC/SPH by local Knudsen number (device helpers header-inline;
per-cell kernels are device lambdas confined to the non-virtual
`encode_cells`/`decode_cells`), and `DeepLearningCodec` is the
learned-classification variant. Class and builder definitions live in
`src/atlas/codec/*.cu`.

### `collider/`
Boundary interaction and collision-aware motion. `Collider` (float,
non-template) owns the surface-interaction kernels and borrows its boundary
units from the `Universe` (`universe.collider_units()`, an `atlas::UnitField`);
`ColliderProbe` is a device-visible view struct, and `HitCollider`
(`collider/hit_collider.h`) is its closest-hit record. Subdirectories:
`collider/interaction/` (surface models such as isothermal and Maxwellian and
the `SurfaceInteractionKernel` DeviceVariant union) and `collider/kernel/`
(the header-inline `ColliderCollisionKernel` sweep plus the post-collision
motion policies: fast, dt-remain, precise, and the `PostColliderKernel`
DeviceVariant union); the per-unit/scene broad-phase AABB cache lives in
`UnitField` under `unit/`. Probe assembly is inlined into
`Collider::make_probe`. Host definitions live in `src/atlas/collider/`
(`collider.cu`, `interaction/*.cu`).

### `solver/`
Physics time-stepping (float, non-template). `Solver` is the host-only
abstract base owning protected `_universe`, `_fluid`, and `_searcher`, with
`solve(dt)` and codec-aware `solve(allocated_solver, index, dt)` virtuals.
Subdirectories:
- `solver/dsmc/` — `DsmcSolver` and variants (`DsmcSimpleSolver`,
  `DsmcEnergyExchangeSolver`), with the HS/VHS/VSS collision kernels
  (`DsmcKernel` DeviceVariant union) and collision statistics.
- `solver/sph/` — `SphSolver` and smoothing kernels
  (cubic spline, standard, Wendland quintic; `SphKernel` DeviceVariant union).

Collision/SPH kernels and the `DsmcProbe`/`SphProbe` view
structs are device-callable header-inline; host-only solver/statistics/probe-
builder definitions live in `src/atlas/solver/{dsmc,sph}/*.cu`. DSMC
solvers cache a `DsmcSolverProbe` in `_probe`; SPH solvers cache an
`SphSolverProbe`. Refresh probe data with `make_probe()` before device launches
when needed.

---

## 5.5 Simulation Objects

### `fluid/`
`Fluid` and the `FluidState` family (float, non-template; converted). Particle
storage: `DeviceBuffer<MaterialProperties>` properties,
`DeviceBuffer<Generate>` generators, and typed per-particle states in a
`TypeStore<FluidState>` (position/velocity as `DeviceBuffer<Float3>`, species
as `DeviceBuffer<std::size_t>`, active as `DeviceBuffer<int>`, temperature as
`DeviceBuffer<float>`, internal energy as `DeviceBuffer<FluidInternalEnergy>`).
The template `compact_buffer`/`reset_buffer` helpers stay inline in the
`FluidState` base; host-only definitions live in `src/atlas/fluid/`
(`fluid.cu`, `fluid_state.cu`). `Fluid::save` / `Builder::with_binary` are
implemented on top of the `serialization/` snapshot functions. See
[02-runtime-objects.md](02-runtime-objects.md).

### `universe/`
`Universe` and the `UniverseState` family (float, non-template; converted).
Domain extents, grid resolution (`Int3`), and per-cell field states stored
in a `TypeStore<UniverseState>`. Per-cell buffers are `DeviceBuffer<float>`,
`DeviceBuffer<Float3>`, or `DeviceBuffer<int>` (collision count);
`UniverseMaterialRatioState<N>` keeps its dimension template and stores
`Container<float, N>`. The `Universe` also centrally owns the scene's units as
one `UnitField` per consumer role (`with_{source,sink,collider,measurer}_units`
builders / `{...}_units()` accessors) — the roles borrow these rather than
owning units. Host-only definitions live in `src/atlas/universe/`
(`universe.cu`, `universe_state.cu`). `Universe::save` / `Builder::with_binary`
are implemented on top of the `serialization/` snapshot functions.

### `source/`
`Source` — particle injection into the inactive tail (float, non-template;
converted). `Spawn` (surface/volume `DeviceTypeSwitch` variant) and the
`SourceProbe` pointer-view struct are header-inline device code. `Source` and
its `Builder`, together with the spawn-cache rebuild, species shuffle, emission
kernels, and probe assembly, are all defined in `src/atlas/source/source.cu`.

### `sink/`
`Sink` — particle removal and active-prefix compaction (float, non-template).
The `Despawn` tag dispatcher and the stateless
volume/surface/tracing despawn operators are header-inline device code;
`SinkProbe` is a device-visible view struct. The unit-bounds refresh and probe
assembly are inlined into `Sink` (`refresh_unit_bounds`, `make_probe`). Host
definitions live in `src/atlas/sink/sink.cu`.

---

## 5.6 Coordination and Driver

### `orchestrator/`
`Orchestrator` (float, non-template) — coordinates the per-step pipeline
(search, classification, measurement, force, solve). `OrchestratorProbe` is a
header-inline float view struct; the stage sequencing, force-application
kernels, and probe assembly are all host code (with device lambdas) in
`src/atlas/orchestrator/orchestrator.cu`. See
[03-simulation-pipeline.md](03-simulation-pipeline.md).

### `system/`
`System` (float, non-template) — top-level step driver. Owns the runtime
objects and exposes `update()`; host definitions (including the
time-integration device lambda) live in `src/atlas/system/system.cu`.

---

## 5.7 Cross-Cutting

### `observer/`
Optional diagnostics. `Observer` holds typed sensor metrics (`SensorMetrics`,
with `SourceSensorMetrics` / `SinkSensorMetrics`) and can export them (for
example to CSV). Host-only; declarations in headers, definitions in
`src/atlas/observer/` (`observer.cu`, `sensor_metrics.cu`).

### `logging/`
Optional `Logger` with severity levels; compiled implementation in
`src/logging/`. Enabled by build configuration.

### `serialization/`
Binary checkpoint/restore (float, non-template). `FluidBinarySnapshot` and
`UniverseBinarySnapshot` plus `save_/load_fluid_binary` and
`save_/load_universe_binary`; the protobuf-backed implementation is compiled
in `src/serialization/` (wire format keeps `double` fields, converted at the
boundary).

---

## 5.8 Python Bindings (Optional)

The Python bindings are *not* part of the engine core under
`include/atlas/`; they live in `src/python/atlas/` and are compiled only when
`ATLAS_PYTHON` is enabled.

- Built with nanobind (vendored under `external/nanobind`).
- `src/python/atlas/module.cpp` defines the importable `atlas` extension module,
  linking `atlas::core` so it shares the same headers and backend macros.
- The current surface is a minimal scaffold (`__version__`, `backend()`, and a
  sample `Vector3D` type), intended to grow incrementally.

---

## 5.9 Umbrella Header

`include/atlas/atlas.h` aggregates every public header. It is generated by
`tools/generate_headers.py`, which walks `include/atlas/`, collects every `.h`
(and `.cuh`) file except the project umbrella and `math.h`, and writes the
sorted `#include` list. Do not hand-edit the umbrella header; regenerate it with
the tool when adding or moving public headers.
