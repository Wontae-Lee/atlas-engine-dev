# CLAUDE.md — Atlas Engine

This file is the working agreement for Claude (Claude Code) when contributing
to the Atlas Engine repository. Read it before making changes.

## Architecture Reference

* The authoritative description of the program structure lives in
  [`docs/architecture/`](docs/architecture/). Read it before changing core code
  under `include/atlas/`.
* Treat `docs/architecture/` as the source of truth for the layered design,
  runtime objects, the per-step pipeline, the backend/portability model, the
  module map, and the structural conventions.
* Base code modifications on the architecture documents: keep the layer
  boundaries, ownership, pipeline order, and naming/file conventions described
  there intact.
* When the user requests an intentional architectural change, update the
  affected document under `docs/architecture/` in the same change so the
  description stays accurate. Do not let the docs and the code drift apart.
* Start points by topic:
  * Layered design and source-tree map → `docs/architecture/01-overview.md`
  * Runtime objects and ownership → `docs/architecture/02-runtime-objects.md`
  * Per-step control flow → `docs/architecture/03-simulation-pipeline.md`
  * CUDA/TBB backend model → `docs/architecture/04-backend-portability.md`
  * Directory-by-directory catalog → `docs/architecture/05-module-map.md`
  * Conventions (files, builders, naming) → `docs/architecture/06-conventions.md`

## General Instructions

* Respond in Korean unless the user asks for another language.
* Do not write source-code comments unless the user explicitly asks for them. When requested, write source-code comments in English.
* Make changes bold enough to fully satisfy the requested behavior. Do not preserve broken structure just to keep a diff small.
* Keep edits focused on the requested behavior, but do not treat minimal line count as a goal.
* Preserve existing style, naming, include order, file layout, and backend portability.
* Do not introduce broad refactors, public API changes, new dependencies, build-system changes, or formatting-only churn unless explicitly requested.
* Be cautious only when the user asks for a conservative change, when public APIs or cross-module contracts would change, or when the invariant needed for a larger fix is unclear.
* Avoid changes that predictably break builds or leave declarations and definitions inconsistent. If the correct fix requires touching related files, update them together.
* Prefer project abstractions and existing invariants over ad-hoc workarounds.
* When a request depends on missing context or an unclear invariant, explain the gap instead of guessing or adding unrelated safeguards.
* When the user asks for `git commit`, group all staged changes except `.idea/workspace.xml` by related purpose, then create commits that match those groups.
* Commit `.idea/workspace.xml` only when the user asks for `git commit all`, and only after all other grouped commits are complete.

## Coding Style

* Implement only the requested algorithm or behavior.
* Prefer structures and patterns that experienced C++ programmers would immediately recognize: clear ownership boundaries, direct control flow, cohesive classes, paired declaration/definition files, and role-named helpers.
* Follow the nearest sibling module's organization before inventing a new layout. Keep helper types close to the owner they support, and split them out only when the role is substantial and named clearly, such as a kernel, builder, probe builder, interaction, or policy.
* Keep code concise and direct; avoid unnecessary temporary variables and redundant branches.
* Prefer concise class and function names, but optimize for readable control flow over raw name length.
* Keep member function names short and natural when the class context already supplies meaning. Prefer names like `apply_collision` over overly explicit names such as `accept_and_scatter_pair` or `particle_index_at_offset`.
* In class declarations and definitions, keep member functions ordered as constructors/destructor, the class's core functions, setters, then getters.
* Keep member variables ordered by type when no nearer sibling layout is more specific: `bool`, integer, floating-point, `HostBuffer<T>`, then `DeviceBuffer<T>`.
* Function names should clearly distinguish each algorithmic step. Avoid near-duplicate names that differ only by a generic suffix or repeated verb, such as `execute_*_trial` and `execute_*_pair`, when more specific step names would make the call flow easier to scan.
* Do not split code into many tiny helpers just to shorten individual functions. A readable implementation should make the algorithmic flow understandable at the call site.
* Do not add defensive checks, fallback paths, ownership guards, recovery branches, diagnostic-only state, or debug scaffolding unless requested or necessary to preserve an existing local contract.
* Do not silently repair invalid states by resetting, zeroing, clamping, skipping required work, or mutating unrelated data unless that is part of the requested algorithm.
* Avoid mutating shared solver, universe, fluid, or searcher state from guard branches. Update only the state owned by the requested algorithmic step.
* Keep temporary logging, counters, assertions, probes, timing code, and instrumentation-only fields out of production code unless requested.
* If the user explicitly asks for source-code comments, use Doxygen-style comments for public APIs or files that already follow that convention.

## Program Structure

The program structure is documented in full under
[`docs/architecture/`](docs/architecture/). That directory is the source of
truth; the points below are a quick reference only. When a detail here and the
architecture docs disagree, the architecture docs win — fix the discrepancy.

* Core headers are header-only under `include/atlas/`. Compiled sources live in
  `src/logging/`, `src/serialization/`, and `src/vizkit/`. The umbrella header
  `include/atlas/atlas.h` is generated by `tools/generate_headers.py` and must
  not be hand-edited. — see `docs/architecture/01-overview.md`.
* Runtime objects: `Fluid<T>` (particles), `Universe<T>` (domain),
  `Source<T>` (emit), `Sink<T>` (remove + compact), `Collider<T>` (boundary
  interaction), `Orchestrator<T>` (per-step coordination), and `System<T>`
  (top-level driver). — see `docs/architecture/02-runtime-objects.md`.
* Per-step flow: `System<T>::update()` runs `emit() -> orchestrate() ->
  advect() -> remove()`; the orchestrator pipeline runs `search ->
  classification -> measurement -> force -> solve`. Without a collider,
  `advect()` falls back to `position += velocity * dt`. — see
  `docs/architecture/03-simulation-pipeline.md`.
* Backend model: exactly one of `ATLAS_USE_CUDA` / `ATLAS_USE_TBB`. Prefer
  `DeviceBuffer<T>`, `HostBuffer<T>`, `device_shared_ptr<T>`,
  `parallel_for<ExecutionPolicy>(...)`, and the macros in `core/macros.h` over
  backend-specific code; keep backend paths behind `ATLAS_TASKING_CUDA` /
  `ATLAS_TASKING_TBB`. — see `docs/architecture/04-backend-portability.md`.
* Module catalog (all directories under `include/atlas/`) — see
  `docs/architecture/05-module-map.md`.

### Structural Guardrails

These follow from the architecture and must hold even on small changes:

* Active-prefix discipline: only `[0, particle_count)` is active; source appends
  behind it; sink compacts survivors and updates `particle_count`. Set capacity
  with `Fluid<T>::Builder::with_buffer_size(...)`, set `particle_count`
  explicitly when active particles should exist, and never assume
  `particle_count == buffer_size`.
* Solver code is performance-sensitive. `Solver<T>` owns protected `_universe`,
  `_fluid`, and `_searcher`; reuse them. Allocation filtering must only skip
  cells outside the selected allocation and must not clear statistics or state
  owned by another solver. Avoid behavior, memory-layout, or ownership changes
  unless requested.
* Builders use the `Collider<T>` style (`class Builder;` inside the owner,
  defined after the owner body in the same header, with a static `builder()`
  factory). `Source<T>::Builder` and `Sink<T>::Builder` use `with_units(...)`;
  `System<T>::Builder` does not own capacity — capacity belongs to `Fluid<T>`.
* File layout: `#pragma once`; paired `.h`/`.hpp` with the `.h` including the
  `.hpp` at the bottom; `.cu` for CUDA TUs and `.cpp` for CPU TUs; keep
  declarations and definitions synchronized.
* Do not silently rename existing public spellings such as `MatrialProperties`
  or `sensor_matrics` unless the task explicitly requests a rename.

## Vizkit

* Vizkit is compiled only when `ATLAS_ENABLE_VIZKIT` is defined.
* `vizkit::Viewer<T>` stores a `SystemHostPtr<T>` and reads `system()->dt()`.
* `Viewer<T>::Builder` uses `.with_system(...)`, `.with_size(...)`, `.with_title(...)`, and `.with_fullscreen(...)`.
* Layer updates receive `(GLFWwindow*, Camera&, T dt)`.

## Build and Test Policy

Do not run builds, tests, benchmarks, simulations, generators, or formatters unless the user explicitly asks.

If the user asks to run project Python tools, activate the virtual environment first:

```bash
source .venv/bin/activate
```

`CMakePresets.json` requires CMake 3.20+ and Ninja. Important presets are:

* Configure: `tbb-debug`, `tbb-release`, `cuda-debug`, `cuda-release`, `cuda-debug-tests`
* Build: `build-tbb-debug`, `build-tbb-release`, `build-cuda-debug`, `build-cuda-release`, `build-cuda-debug-tests`
* Test: `ctest-tbb-debug`, `ctest-tbb-release`

Important options:

* `ATLAS_USE_CUDA`, `ATLAS_USE_TBB`
* `ATLAS_USE_VIZKIT`, `ATLAS_LOGGING`
* `ATLAS_GOOGLE_TEST`, `ATLAS_CUDA_TEST`, `ATLAS_BENCHMARKS`

Constraints:

* Exactly one of `ATLAS_USE_CUDA` and `ATLAS_USE_TBB` must be enabled.
* `ATLAS_GOOGLE_TEST` is disabled for CUDA presets.
* Benchmarks are currently TBB-only.

## Test Guidelines

* Shared include: `src/testkit/testkit.h`
* C++ tests use GoogleTest through `testkit`; CUDA tests use the in-tree `cudatest` through `testkit`.
* Include `<testkit/testkit.h>` instead of GoogleTest or cudatest headers directly.
* Keep shared helpers in `tests/utilities/test_utils.h`.
* Place tests under `tests/<module>/` to match the public module or runtime subsystem.
* Name C++ test files `<subject>_tests.cpp` and CUDA companion files `<subject>_tests.cu`.
* Keep local aliases, helper functions, and fixtures in an anonymous namespace.
* Prefer focused `TEST(SuiteName, BehaviorName)` cases that describe observable behavior.
* Do not add another C++ or CUDA test `main`; existing entry points are provided by `testkit` and `tests/cuda/main.cu`.

Key test locations:

* `tests/system/system_tests.cpp`
* `tests/source/`
* `tests/sink/`
* `tests/observer/`
* `tests/material/`
* `tests/solver/`

## Dependencies and Reference Code

External dependencies include TBB, CUDA 12.x, and the OpenGL stack used by Vizkit. In-tree dependencies under `external/` include tinyobjloader, Lyra, googletest, googlebenchmark, and protobuf.

Benchmark reference submodules live under `benchmarks/`:

* `benchmarks/dumux`
* `benchmarks/piclas`
* `benchmarks/sparta`
* `benchmarks/splishsplash`

When the user mentions `piclas`, `dumux`, `sparta`, or `splishsplash` in benchmark or reference-code context, treat the name as referring to the corresponding submodule directory.
