# AGENTS.md — Atlas Engine

## General AI Instructions

* Respond in Korean unless the user asks for another language.
* Write source-code comments in English.
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
* Keep code concise and direct; avoid unnecessary temporary variables, redundant branches, and verbose comments.
* Prefer concise class and function names, but optimize for readable control flow over raw name length.
* Keep member function names short and natural when the class context already supplies meaning. Prefer names like `apply_collision` over overly explicit names such as `accept_and_scatter_pair` or `particle_index_at_offset`.
* Function names should clearly distinguish each algorithmic step. Avoid near-duplicate names that differ only by a generic suffix or repeated verb, such as `execute_*_trial` and `execute_*_pair`, when more specific step names would make the call flow easier to scan.
* Do not split code into many tiny helpers just to shorten individual functions. A readable implementation should make the algorithmic flow understandable at the call site.
* Do not add defensive checks, fallback paths, ownership guards, recovery branches, diagnostic-only state, or debug scaffolding unless requested or necessary to preserve an existing local contract.
* Do not silently repair invalid states by resetting, zeroing, clamping, skipping required work, or mutating unrelated data unless that is part of the requested algorithm.
* Avoid mutating shared solver, universe, fluid, or searcher state from guard branches. Update only the state owned by the requested algorithmic step.
* Keep temporary logging, counters, assertions, probes, timing code, and instrumentation-only fields out of production code unless requested.
* Use Doxygen-style comments for public APIs or files that already follow that convention.

## Project Overview

Atlas is a C++20 particle simulation engine with a mostly header-only core.

* Core headers: `include/atlas/`
* Optional OpenGL visualization: `src/vizkit/`
* Compiled sources: `src/logging/`, `src/vizkit/`, `src/serialization/`
* Generated umbrella header: `include/atlas/atlas.h`
* Do not hand-edit generated umbrella headers; use `tools/generate_headers.py`.

## Backend and Portability

Exactly one backend must be enabled: `ATLAS_USE_CUDA` or `ATLAS_USE_TBB`.

* Backend macros: `ATLAS_TASKING_CUDA`, `ATLAS_TASKING_TBB`
* `DeviceBuffer<T>` maps to `thrust::device_vector<T>` on CUDA and `std::vector<T>` on TBB.
* `HostBuffer<T>` maps to `thrust::host_vector<T>` on CUDA and `std::vector<T>` on TBB.
* `device_shared_ptr<T>` maps to CUDA-aware ownership or `std::shared_ptr<T>`.
* Prefer `parallel_for<ExecutionPolicy>(...)` and project abstractions over backend-specific code.

Use portability macros from `include/atlas/core/macros.h` when appropriate:

* `ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE`
* `ATLAS_FORCE_INLINE`, `ATLAS_NODISCARD`, `ATLAS_MAYBE_UNUSED`
* `RESTRICT`

## Runtime Architecture

* `atlas::Fluid<T>` owns particle storage, material/species properties, generator operators, and fluid states.
* `atlas::Universe<T>` owns domain extents, grid resolution, and universe states.
* `atlas::Source<T>` emits particles into inactive capacity after the active prefix.
* `atlas::Sink<T>` removes particles and compacts survivors into the active prefix.
* `atlas::system::System<T>` is the high-level simulation driver.

`System<T>::update()` runs:

```text
source->update(dt)
-> orchestrator->update(dt)
-> collider->update(dt) or time_integration()
-> sink->update(dt)
```

Collision-aware motion is handled through `Collider<T>` and `System<T>::advect()`. Without a collider, `System<T>::time_integration()` performs `position += velocity * dt`.

## Active Prefix Discipline

* `Fluid<T>::buffer_size()` is total allocated capacity.
* `Fluid<T>::particle_count()` is the active dense prefix length.
* Only `[0, particle_count)` is active.
* Source emission appends behind the active prefix.
* Sink removal compacts active particles and updates `particle_count`.

When writing setup code or tests, set capacity with `Fluid<T>::Builder::with_buffer_size(...)`, set `particle_count` explicitly when active particles should exist, and never assume `particle_count == buffer_size`.

## Solver Guidelines

* `Solver<T>` owns protected `_universe`, `_fluid`, and `_searcher`; derived solvers should reuse these members instead of duplicating dependencies.
* DSMC solvers use `DsmcSolver<T>::DsmcSolverProbe` cached in `_probe`; call `make_probe()` before device launches when probe data must be refreshed.
* SPH solvers use `SphSolver<T>::SphSolverProbe` cached in `_probe`.
* Solver-allocation filtering should be passed as function parameters, not stored in probes.
* Allocation filtering must only skip cells outside the selected allocation. It must not clear collision statistics or universe state owned by another solver.
* Solver code is performance-sensitive; avoid behavior, memory-layout, or ownership changes unless requested.

## Orchestrator and System

* `atlas::system::Orchestrator<T>` coordinates optional search, codec, measurement, field-force/gravity application, and solvers.
* `System<T>::Builder::with_solver(...)` currently accepts an `OrchestratorHostPtr<T>`.
* The standalone `Advector` API and `include/atlas/advector/` module are no longer used.

## Vizkit

* Vizkit is compiled only when `ATLAS_ENABLE_VIZKIT` is defined.
* `vizkit::Viewer<T>` stores a `SystemHostPtr<T>` and reads `system()->dt()`.
* `Viewer<T>::Builder` uses `.with_system(...)`, `.with_size(...)`, `.with_title(...)`, and `.with_fullscreen(...)`.
* Layer updates receive `(GLFWwindow*, Camera&, T dt)`.

## Namespaces and Builders

* Common aliases are re-exported under `atlas::`.
* Runtime types mostly live under `atlas::system::`, `atlas::fluid::`, `atlas::universe::`, and `atlas::geometry::`.
* Math types live under `atlas::math::`, with convenience aliases in `atlas::`.
* Most public types use nested `Builder` classes with `.with_*()` setters, `.build()`, and `.make_host_shared()`.
* Write nested builders in the `Collider<T>` style: declare `class Builder;` inside the owning class, then define `template <typename T> class Type<T>::Builder final` after the owning class body in the same header. Keep `builder()` as a static factory on the owning class.
* `Source<T>::Builder` and `Sink<T>::Builder` use `with_units(...)`.
* `System<T>::Builder` does not own particle capacity; capacity belongs to `Fluid<T>`.

## File and Module Conventions

* Use `#pragma once` in headers.
* Template components use paired `.h` and `.hpp` files in the same directory, with the `.h` including the `.hpp` at the bottom.
* Keep declarations and definitions synchronized.
* CUDA translation units use `.cu`; CPU translation units use `.cpp`.
* Do not silently rename existing public spellings such as `MatrialProperties` or `sensor_matrics` unless the task explicitly requests a rename.
* `include/atlas/system/system.h` and `include/atlas/system/system.hpp` should keep this order: constructors/destructor, major runtime functions, setters, getters, clear functions.

Current `include/atlas/` modules include:

```text
atomic, buffer, codec, collider, container, core, fluid, generator,
geometry, indexer, iterator, logging, material, math, measure,
memory, observer, orchestrator, parallel, random, remove, sampling,
scan, searcher, serialization, shuffle, sink, solver, source,
spatial, sync, system, transform, tuple, unit, universe
```

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
