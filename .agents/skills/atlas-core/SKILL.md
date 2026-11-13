---
name: atlas-core
description: Implement, review, or document Atlas computational Core modules in include/atlas and src/atlas, preserving builder semantics, ownership, and shared TBB/CUDA execution.
---

# Atlas Core work

Apply the repository [bootstrap](../../SKILL.md) first. Use this skill for Core
behavior, public headers, and their module documentation. Frontend command
handling and example presentation belong to their respective layers.
Apply the common [coding style](../atlas-coding-style/SKILL.md) before editing;
this skill adds Core architecture and integration guidance.

## Locate the owner

Read the affected header, its implementation when present, and the matching
[Core module document](../../../docs/atlas/README.md). Not every header has a
translation unit: math, buffer aliases, and dispatch machinery are largely
header-defined. Most compiled engine implementations use `.cu`, even for TBB;
[AtlasLibraries.cmake](../../../cmake/AtlasLibraries.cmake) compiles those sources
as C++ in the native build.

| Work | Source and documentation to inspect |
|---|---|
| Particle columns, capacity, compaction | [Fluid](../../../include/atlas/fluid/fluid.h), [FluidState](../../../include/atlas/fluid/fluid_state.h), `src/atlas/fluid/`, `docs/atlas/fluid/` |
| Grid and cell fields | `include/atlas/universe/`, `src/atlas/universe/`, `docs/atlas/universe/`; search membership also uses `searcher/` |
| Geometry and rigid motion | [Unit](../../../include/atlas/unit/unit.h), [Unit builder](../../../src/atlas/unit/unit.cu), `sync/`, `geometry/`, and their module documents |
| Emission, removal, collisions | The corresponding `source/`, `generator/`, `sink/`, `collider/`, or `solver/` headers, sources, and module documents |
| Pipeline phase ordering | [System implementation](../../../src/atlas/system/system.cu), [pipeline contract](../../../docs/architecture/simulation-pipeline.md), `docs/atlas/system/` |
| Backend buffers and dispatch | `include/atlas/buffer/`, `memory/`, `parallel/`, `core/`, and their module documents |
| Persistence | `include/atlas/serialization/`, `src/atlas/serialization/`, [serialization guide](../../../docs/atlas/serialization/serialization.md) |

Paths in the table without links are repository-relative. Follow neighboring
implementations for the affected family rather than applying one class layout
to every module.

## Extend values and builders together

The [VolumeSink header](../../../include/atlas/sink/volume_sink.h) and
[implementation](../../../src/atlas/sink/volume_sink.cu) show the common split:
small device-callable value operations in the header; host builder construction,
validation, and ownership transfer in the `.cu`. The leaf declares its nested
`Builder`, whose definition follows the leaf. Follow the existing annotations,
private member naming, return-type placement, and Doxygen style in that family.

When adding state, trace the complete path: member defaults, constructors,
builder setters, `validate()`, `build()`, and any builder reset. Preserve optional
state as optional when absence has behavior distinct from zero. A constructor
must initialize the same state that `build()` passes. Builder consumption is
class-specific: `Unit` and `VolumeSink` reset staged members, while other builders
may support repeated const builds; retain the existing contract.

Use [math constants and helpers](../../../include/atlas/math/constants.h),
`Float3`, and existing `Sync`/`Unit` pose operations before introducing duplicate
math. Existing motion state is float-based; do not add double intermediates and
casts without a numerical reason. Keep domain validation on the host builder
path, not inside device operations that cannot report exceptions.

## Keep ownership distinct from device access

- `Fluid` owns capacity-sized columns; `particle_count()` selects the live
  prefix. New particle attributes must participate in the common compaction
  path so all columns retain the same particle ordering.
- `DeviceBuffer` and `HostBuffer` own storage. Kernel captures use raw pointers
  or lightweight views obtained through the memory helpers, not owning buffers
  or host smart pointers. Check the owner lifetime and invalidation after resize.
- [Source](../../../include/atlas/source/source.h) uses `HostVariant` because its
  leaves own buffers and move on the host. [Sink](../../../include/atlas/sink/sink.h)
  uses `DeviceVariant` for copyable device leaves. Extending a family also means
  updating its tag, union, variant case list, concept assertions, and forwarding
  operations; do not choose a variant solely to match another class's spelling.
- [TriangleMesh](../../../include/atlas/geometry/triangle_mesh.h) owns triangles,
  BVH, and query caches; `TriangleMeshView` and `Geometry` borrow that storage.
  Keep the mesh owner alive wherever a view is stored. Use the existing OBJ
  loader and supported cache-refresh operations when editing mesh data.

Use [parallel_for](../../../include/atlas/parallel/parallel_for.h) with the
appropriate execution policy. `ExecutionPolicy::device` means CUDA device work
in the CUDA build and host parallel work in TBB. Lambdas on that shared path
must retain the host/device annotations and capture only suitable values. Do
not infer that a `.cu` suffix requires a CUDA-only implementation.

## Preserve the step contract

`System::update()` runs `emit → search → allocate → solve → advect → remove`,
then increments the step. Search establishes membership consumed by solving;
removal compacts columns and invalidates old indices. During advection, particles
trace against start-of-step collider poses; collider Units advance afterward in
a separate pass. Keep motion and wall-reflection state consistent rather than
animating only a frontend copy.

For changed public state or physics, inspect affected constructors and callers,
module documentation, and the relevant tests before deciding what else needs
updating. Persistence and frontend adapters need changes only when the requested
API or stored behavior crosses those boundaries. The bootstrap's execution
restrictions still govern builds, tests, and generated headers; this skill does
not authorize running them.
