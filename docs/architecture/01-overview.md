# 1. Architecture Overview

Atlas is a C++20 engine for particle-based physics simulation, compiled with a
single toolchain (nvcc). It targets two execution backends from one source
tree — Intel TBB for CPU parallelism and NVIDIA CUDA for GPU parallelism —
selected through the Thrust device system, and keeps most runtime APIs
backend-agnostic. The engine uses a single scalar type: `float`.

This document describes the layered design and where each layer lives in the
source tree. Later documents drill into the runtime objects, the per-step
pipeline, the backend model, the full module map, and the conventions.

---

## 1.1 Design Goals

The architecture is shaped by a small number of recurring goals:

```text
- One source tree compiles for either TBB or CUDA (Thrust device system),
  always through nvcc.
- The scalar type is float; public types are plain non-template classes.
- The hot path stays branch-light and kernel-inlinable (header-inline device
  code); host-only logic compiles into the atlas library (src/atlas/*.cu).
- Runtime objects own their data and are assembled through builders.
- Physics models (solvers, collision kernels, geometries) are pluggable.
- Per-particle storage uses a dense active prefix, not a fixed population.
```

---

## 1.2 Layered Design

The core is organized as a stack of layers. Higher layers depend on lower
layers, not the other way around.

```text
+--------------------------------------------------------------+
| Driver layer                                                 |
|   System<T>                                                  |
|   - top-level step driver: emit -> orchestrate -> advect ->  |
|     remove                                                   |
+--------------------------------------------------------------+
| Coordination layer                                           |
|   Orchestrator<T>                                            |
|   - search -> classify -> measure -> apply forces -> solve   |
+--------------------------------------------------------------+
| Simulation-object layer                                      |
|   Fluid<T>, Universe<T>, Source<T>, Sink<T>, Collider<T>     |
+--------------------------------------------------------------+
| Physics layer                                                |
|   Solver<T> (DSMC, SPH, hybrid), collision/interaction       |
|   kernels, codec, measurer, generator                        |
+--------------------------------------------------------------+
| Spatial layer                                                |
|   Searcher<T>, indexer, geometry, spatial (AABB, ray, BVH),  |
|   unit, sync, transform                                      |
+--------------------------------------------------------------+
| Primitive layer                                              |
|   math (vector, matrix, quaternion), buffer, memory,         |
|   parallel, scan, remove, sampling, shuffle, random,         |
|   container, tuple, iterator, unit utilities                 |
+--------------------------------------------------------------+
| Portability layer                                            |
|   core/macros.h - host/device attributes, inline, nodiscard  |
+--------------------------------------------------------------+
```

Two cross-cutting concerns sit beside the stack rather than inside it:

```text
- Observability: observer/, logging/ (optional diagnostics and telemetry)
- Persistence:   serialization/ (binary snapshots of Fluid and Universe)
```

---

## 1.3 Source-Tree Map

The repository separates public headers from compiled and optional
components.

```text
include/atlas/      Public headers (declarations + inline device code).
include/atlas/atlas.h
                    Generated umbrella header; do not hand-edit.

src/atlas/          Engine definitions (.cu), mirroring include/atlas/.
                    Compiled by nvcc into the `atlas` static library.
src/logging/        Compiled logging implementation.
src/serialization/  Compiled serialization (protobuf snapshots).
src/python/atlas/   nanobind-based Python bindings (built when
                    ATLAS_PYTHON is enabled).

tests/              GoogleTest sources, mirroring the module layout.
benchmarks/         Benchmark cases (formerly examples/) and
                    reference-code submodules.
tools/              Python helpers, including generate_headers.py.
docs/               Documentation, including this architecture directory.
```

The boundary that matters most for everyday work: **headers under
`include/atlas/` declare the API and hold inline device-callable code;
host-only definitions live in `src/atlas/` as `.cu` translation units that
mirror the header layout.**

---

## 1.4 Backend Model in One Paragraph

Everything is compiled by nvcc; the backend is the Thrust device system chosen
at configure time (`ATLAS_DEVICE_SYSTEM=TBB` for CPU, `CUDA` for GPU), with the
Thrust host system fixed to TBB. Backend differences are hidden behind shared
aliases — `DeviceBuffer<T>`, `HostBuffer<T>`, `device_shared_ptr<T>` — and the
`parallel_for<ExecutionPolicy>(...)` dispatch family. Compile-time backend
paths key off `ATLAS_TASKING_TBB` / `ATLAS_TASKING_CUDA`. Most code should use
the shared abstractions rather than backend-specific calls. See
[04-backend-portability.md](04-backend-portability.md) for details.

---

## 1.5 Where to Go Next

- To understand what the runtime objects own and how they are built, read
  [02-runtime-objects.md](02-runtime-objects.md).
- To follow a single simulation step end to end, read
  [03-simulation-pipeline.md](03-simulation-pipeline.md).
- To catalog every module directory, read [05-module-map.md](05-module-map.md).
