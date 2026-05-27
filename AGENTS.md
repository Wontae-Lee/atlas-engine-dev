# AGENTS.md — Atlas Engine Dev

## Codex Operating Rules

- Communicate with the user in Korean unless they ask otherwise.
- Write source-code comments in English.
- Keep changes small and scoped to the user's request.
- Do not refactor broadly, change public APIs, or add dependencies unless explicitly asked.
- Preserve existing style, naming, include order, file layout, and backend portability.
- Do not add defensive checks, fallback paths, or new validation unless requested or already required by the local pattern.
- Do not run builds, tests, benchmarks, simulations, or formatters unless the user explicitly asks. If not run, state that clearly.

## Project Overview

Atlas is a C++20 particle simulation engine with a mostly header-only core.

- Core headers: `include/atlas/`
- Optional OpenGL visualization: `src/vizkit/`
- Compiled sources: `src/logging/`, `src/vizkit/`, `src/serialization/`
- Generated umbrella header: `include/atlas/atlas.h`
- Do not hand-edit generated umbrella headers; use `tools/generate_headers.py`.

## Backend Model

- Exactly one backend must be enabled: `ATLAS_USE_CUDA` or `ATLAS_USE_TBB`.
- Backend macros are `ATLAS_TASKING_CUDA` and `ATLAS_TASKING_TBB`.
- `DeviceBuffer<T>` is `thrust::device_vector<T>` on CUDA and `std::vector<T>` on TBB.
- `HostBuffer<T>` is `thrust::host_vector<T>` on CUDA and `std::vector<T>` on TBB.
- `device_shared_ptr<T>` maps to CUDA-aware ownership or `std::shared_ptr<T>`.
- Use `parallel_for<ExecutionPolicy>(...)` and project abstractions instead of backend-specific code when possible.

Use portability macros from `include/atlas/core/macros.h`:

- `ATLAS_HOST`
- `ATLAS_DEVICE`
- `ATLAS_ALL_DEVICE`
- `ATLAS_FORCE_INLINE`
- `ATLAS_NODISCARD`
- `ATLAS_MAYBE_UNUSED`
- `RESTRICT`

## Runtime Architecture

- `atlas::Fluid<T>` owns particle storage, material/species properties, generator operators, and fluid states.
- `atlas::Universe<T>` owns domain extents, grid resolution, and universe states.
- `atlas::Source<T>` emits particles into inactive capacity after the active prefix.
- `atlas::Sink<T>` removes particles and compacts survivors back into the active prefix.
- `atlas::system::System<T>` is the high-level simulation driver.

`System<T>::update()` runs:

```text
source->update(dt)
-> orchestrator->update(dt)
-> collider->update(dt) or time_integration()
-> sink->update(dt)
```

Collision and motion:

- The old standalone `Advector` API is gone.
- There is no `include/atlas/advector/` module.
- Collider-aware motion is handled through `Collider<T>` and `System<T>::advect()`.
- Without a collider, `System<T>::time_integration()` performs `position += velocity * dt`.

## Active Prefix Discipline

- `Fluid<T>::buffer_size()` is total allocated capacity.
- `Fluid<T>::particle_count()` is the active dense prefix length.
- Only `[0, particle_count)` is active.
- Source emission appends behind the active prefix.
- Sink removal compacts active particles and updates `particle_count`.

When writing tests or setup code:

1. Set capacity through `Fluid<T>::Builder::with_buffer_size(...)`.
2. Set `particle_count` explicitly when active particles should exist.
3. Never assume `particle_count == buffer_size`.

## Solver Structure

- `Solver<T>` owns protected `_universe`, `_fluid`, and `_searcher` dependencies. Derived solvers should use the base storage, not duplicate these members.
- DSMC solvers use `DsmcSolver<T>::DsmcSolverProbe` cached in `_probe`.
- `DsmcSolver<T>::make_probe()` refreshes `_probe`; derived DSMC solvers copy `this->_probe` before device launches.
- `DsmcSolver<T>` handles DSMC state setup, no-time-counter statistics, and stochastic collision-count rounding.
- DSMC allocation filtering is passed as function parameters, not stored in the probe.
- `SphSolver<T>` uses `SphSolver<T>::SphSolverProbe` cached in `_probe`.
- `SphSolver<T>::make_probe()` stores runtime particle/search/universe pointers in `_probe`.
- `SphSolverProbe` does not store solver-allocation data.
- `SphGatewaySolver<T>` caches an SPH probe in protected `_probe` and uses function parameters for allocation filtering.
- Solver code is performance-sensitive; avoid behavior or memory-layout changes unless requested.

## Orchestrator

- `atlas::system::Orchestrator<T>` coordinates optional search, codec, measurement, field-force/gravity application, and solvers.
- `System<T>::Builder::with_solver(...)` currently accepts an `OrchestratorHostPtr<T>`.

## Vizkit

- Vizkit is compiled only when `ATLAS_ENABLE_VIZKIT` is defined.
- `vizkit::Viewer<T>` stores a `SystemHostPtr<T>`.
- `Viewer<T>` reads `system()->dt()`; it does not own a separate timestep.
- `Viewer<T>::Builder` uses `.with_system(...)`, `.with_size(...)`, `.with_title(...)`, and `.with_fullscreen(...)`.
- Layer updates receive `(GLFWwindow*, Camera&, T dt)`.

## Namespaces

- Common aliases are re-exported under `atlas::`.
- Runtime types mostly live under `atlas::system::`, `atlas::fluid::`, `atlas::universe::`, and `atlas::geometry::`.
- Math types live under `atlas::math::`, with convenience aliases in `atlas::`.

## Builders

Most public types use nested `Builder` classes with `.with_*()` setters, `.build()`, and `.make_host_shared()`.

Important details:

- `Source<T>::Builder` uses `with_units(...)`, not `with_unit(...)`.
- `Sink<T>::Builder` uses `with_units(...)`, not `with_unit(...)`.
- `System<T>::Builder` does not own particle capacity; capacity belongs to `Fluid<T>`.

## Template Files

- Template components use paired `.h` and `.hpp` files in the same directory.
- The `.h` includes the `.hpp` at the bottom.
- Keep declarations and definitions synchronized.

`include/atlas/system/system.h` and `include/atlas/system/system.hpp` should stay ordered as:

1. constructors / destructor
2. major runtime functions
3. setters
4. getters
5. clear functions

## Include Modules

Current `include/atlas/` modules:

- `atomic`, `buffer`, `codec`, `collider`, `container`, `core`
- `fluid`, `generator`, `geometry`, `indexer`, `iterator`
- `logging`, `material`, `math`, `measure`, `memory`, `observer`, `orchestrator`
- `parallel`, `random`, `remove`, `sampling`, `scan`, `searcher`, `serialization`, `shuffle`
- `sink`, `solver`, `source`, `spatial`, `sync`, `system`
- `transform`, `tuple`, `unit`, `universe`

## Python Environment

- Before using Python, activate the project virtual environment with `source .venv/bin/activate`.
- The virtual environment contains the project Python interpreter and required libraries.

## Build Reference

`CMakePresets.json` requires CMake 3.20+. All presets use Ninja.

Do not run these commands unless the user explicitly asks.

```bash
# TBB
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug

# CUDA
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)

# CUDA tests
cmake --preset cuda-debug-tests
cmake --build build/cuda-debug-tests --target atlas_all_cuda_test -j$(nproc)
./build/cuda-debug-tests/atlas_all_cuda_test --gtest_list_tests
```

Configure presets:

- TBB: `tbb-debug`, `tbb-release`
- CUDA: `cuda-debug`, `cuda-release`, `cuda-debug-tests`

Build presets:

- `build-tbb-debug`, `build-tbb-release`
- `build-cuda-debug`, `build-cuda-release`, `build-cuda-debug-tests`

Test presets:

- `ctest-tbb-debug`, `ctest-tbb-release`

Important options:

- `ATLAS_USE_CUDA`
- `ATLAS_USE_TBB`
- `ATLAS_USE_VIZKIT`
- `ATLAS_LOGGING`
- `ATLAS_GOOGLE_TEST`
- `ATLAS_CUDA_TEST`
- `ATLAS_BENCHMARKS`

Constraints:

- Exactly one of `ATLAS_USE_CUDA` and `ATLAS_USE_TBB` must be enabled.
- `ATLAS_GOOGLE_TEST` is disabled for CUDA presets.
- Benchmarks are currently TBB-only.

## Tests

- Shared include: `src/testkit/testkit.h`
- C++ tests use GoogleTest through `testkit`.
- CUDA tests use the in-tree `cudatest` through `testkit`.
- Test discovery uses recursive `tests/*.cpp` and `tests/*.cu` globbing from the root `CMakeLists.txt`.
- Test sources should include `<testkit/testkit.h>`, not GoogleTest headers directly.
- Common helpers live in `tests/utilities/test_utils.h`.

Test binaries:

- Aggregate C++ binary: `atlas_tests`
- Per-directory C++ binaries: `atlas_tests_<directory>`
- CUDA binary: `atlas_all_cuda_test`
- CUDA entry point: `tests/cuda/main.cu`

Focused coverage:

- `System`: `tests/system/system_tests.cpp`
- `Source`: `tests/source/`
- `Sink`: `tests/sink/`
- `Observer`: `tests/observer/`
- `Material`: `tests/material/`
- Solver tests: `tests/solver/`

## File Conventions

- Use `#pragma once` in headers.
- CUDA translation units use `.cu`; CPU translation units use `.cpp`.
- Public APIs should keep Doxygen-style comments where surrounding files use them.
- Keep the existing public alias spelling `MatrialProperties` unless the task explicitly includes a rename.
- Keep existing `sensor_matrics` naming in observer APIs and CSV outputs; do not silently normalize it in isolated edits.

## Dependencies

External:

- **TBB**: TBB backend
- **CUDA 12.x**: CUDA backend, with `--expt-relaxed-constexpr --extended-lambda`
- **OpenGL stack**: Vizkit builds (`glfw3`, `GLEW`, `GLU`, `GLUT`)

In-tree under `external/`:

- **tinyobjloader**: OBJ mesh loading
- **Lyra**: header-only CLI parsing
- **googletest**: C++ unit tests
- **googlebenchmark**: microbenchmarks
- **protobuf**: binary snapshot serialization
