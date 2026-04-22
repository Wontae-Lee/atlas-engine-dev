# AGENTS.md — Atlas Engine Dev

## Project Overview

Atlas is a C++20 particle simulation engine with a header-only core in `include/atlas/` and an optional OpenGL visualization layer in `src/vizkit/`. Most engine behavior is template-based and inline; compiled sources are primarily in `src/logging/` and `src/vizkit/`.

## Architecture

- **Header-only core**: `include/atlas/` contains the simulation, math, geometry, container, sampling, source/sink, and system layers. The umbrella header is `atlas/atlas.h`.
- **Backend abstraction**: exactly one of `ATLAS_TASKING_CUDA` or `ATLAS_TASKING_TBB` must be active.
  - `DeviceBuffer<T>` maps to `thrust::device_vector<T>` on CUDA and `std::vector<T>` on TBB.
  - `HostBuffer<T>` maps to `thrust::host_vector<T>` on CUDA and `std::vector<T>` on TBB.
  - `parallel_for<ExecutionPolicy>(...)` dispatches to Thrust or TBB.
  - `device_shared_ptr<T>` maps to CUDA managed-memory ownership or `std::shared_ptr<T>`.
- **Simulation orchestration**: `atlas::system::System<T>` is now the high-level runtime object.
  - Owns `ParticleData<T>` and exposes one `ParticleDeviceProbe<T>`.
  - Stores simulation `dt`.
  - Stores `HostBuffer<SourceHostPtr<T>>`, `HostBuffer<SinkHostPtr<T>>`, and collider lists.
  - Executes staged updates through `update()`, `emit()`, `advect()`, `remove()`, and `time_integration()`.
- **Particle ownership split**:
  - `ParticleData<T>` owns storage buffers.
  - `ParticleDeviceProbe<T>::particle_count` is the active prefix length.
  - `ParticleDeviceProbe<T>::buffer_size` is the allocated capacity.
- **Source / sink pipeline**:
  - `Source<T>` writes into inactive capacity through `emit(ParticleDeviceProbe<T>&)`.
  - `Sink<T>` compacts the active prefix with `remove_if`.
  - `System<T>::update()` runs `emit() -> advect() -> remove()`.
- **Collision / advection**:
  - The former standalone `Advector` API has been removed.
  - Collider-aware particle motion now lives in `System<T>::advect()`.
  - When no colliders are configured, `advect()` falls back to `time_integration()`.
- **Vizkit** (`src/vizkit/`): optional viewer and layer system behind `#ifdef ATLAS_ENABLE_VIZKIT`.
  - `vizkit::Viewer<T>` no longer owns its own `dt`.
  - `Viewer<T>` stores a `SystemHostPtr<T>` and reads `system()->dt()` during rendering updates.
- **Namespaces**:
  - Core aliases are under `atlas::`.
  - Simulation runtime types are under `atlas::system::`.
  - Geometry is under `atlas::geometry::`.
  - Math is under `atlas::math::` with convenience aliases in `atlas::`.

## Key Patterns

### Builder Pattern
Most public types use a nested `Builder` with fluent `.with_*()` setters, `.build()`, and `.make_host_shared()`:
```cpp
const auto fluid = atlas::system::Fluid<float>::builder()
    .add_species(species_ptr, 1.0f, generator)
    .make_host_shared();

const auto source = atlas::system::Source<float>::builder()
    .with_unit(unit)
    .with_fluid(fluid)
    .with_spacing(0.02f)
    .build();

const auto sim_system = atlas::system::System<float>::builder()
    .with_buffer_size(4096)
    .with_dt(0.01f)
    .with_source(source_ptr)
    .with_sink(sink_ptr)
    .with_collider(collider_ptr)
    .make_host_shared();
```

### Device Portability Macros
Use macros from `include/atlas/core/macros.h` instead of raw CUDA attributes:
- `ATLAS_HOST`
- `ATLAS_DEVICE`
- `ATLAS_ALL_DEVICE`
- `ATLAS_FORCE_INLINE`
- `ATLAS_NODISCARD`
- `ATLAS_MAYBE_UNUSED`
- `RESTRICT`

### Header / Implementation Split
Template components use paired `.h` and `.hpp` files in the same directory. The `.h` includes the `.hpp` at the bottom.

### System API Ordering
`include/atlas/system/system.h` and `include/atlas/system/system.hpp` are intentionally ordered as:
1. constructors / destructor
2. major runtime functions
3. setters
4. getters
5. clear functions

Preserve that ordering when editing `System`.

## Include Layout

Key `include/atlas/` modules currently include:
- `buffer`, `codec`, `collider`, `container`, `core`, `data`, `domain`
- `flatten`, `generator`, `geometry`, `indexer`, `iterator`
- `logging`, `math`, `matter`, `memory`, `parallel`, `random`
- `remove`, `sampling`, `scan`, `searcher`, `shuffle`
- `sink`, `solver`, `source`, `spatial`, `sync`, `system`
- `transform`, `tuple`, `unit`

There is no longer an `include/atlas/advector/` module.

## Vizkit Notes

- `src/vizkit/viewer/viewer.h` depends on `atlas/system/system.h`.
- `Viewer<T>::Builder` uses `.with_system(...)`, `.with_title(...)`, `.with_size(...)`, and `.with_fullscreen(...)`.
- Layer updates still receive `(GLFWwindow*, Camera&, T dt)`, but that `dt` comes from the bound `System`.

## Build & Test

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --test-dir build/tbb-debug --output-on-failure

cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)

cmake --preset tbb-debug-core
cmake --build build/tbb-debug-core -j$(nproc)
ctest --preset ctest-tbb-debug-core

cmake --preset tbb-release-core
cmake --build build/tbb-release-core -j$(nproc)
ctest --preset ctest-tbb-release-core

cmake --preset cuda-debug-core
cmake --build build/cuda-debug-core -j$(nproc)

cmake --preset cuda-debug-tests
cmake --build build/cuda-debug-tests --target atlas_all_cuda_test -j$(nproc)
```

Available configure presets from `CMakePresets.json`:
- TBB: `tbb-debug`, `tbb-release`, `tbb-debug-core`, `tbb-release-core`, `tbb-relwithdebinfo`, `tbb-debug-make`
- CUDA: `cuda-debug`, `cuda-release`, `cuda-debug-core`, `cuda-release-core`, `cuda-debug-tests`, `cuda-relwithdebinfo`

Available build presets:
- `build-tbb-debug`, `build-tbb-release`, `build-tbb-debug-core`, `build-tbb-release-core`, `build-tbb-relwithdebinfo`
- `build-cuda-debug`, `build-cuda-release`, `build-cuda-relwithdebinfo`
- `build-cuda-debug-core`, `build-cuda-release-core`, `build-cuda-debug-tests`

Available test presets:
- `ctest-tbb-debug`, `ctest-tbb-release`, `ctest-tbb-debug-core`, `ctest-tbb-release-core`

Important options:
- `ATLAS_USE_CUDA`
- `ATLAS_USE_TBB`
- `ATLAS_USE_VIZKIT`
- `ATLAS_LOGGING`
- `ATLAS_GOOGLE_TEST`
- `ATLAS_CUDA_TEST`
- `ATLAS_BENCHMARKS`

## Tests

- Shared test include: `src/testkit/testkit.h`
- C++ tests use GoogleTest through `testkit` when `ATLAS_GOOGLE_TEST=ON`
- CUDA tests use the in-tree `cudatest` implementation through `testkit` when `ATLAS_CUDA_TEST=ON`
- Test discovery: recursive `tests/*.cpp` and `tests/*.cu` glob from the root `CMakeLists.txt`
- C++ test binaries:
  - aggregate binary: `atlas_tests`
  - per-directory binaries: `atlas_tests_<directory>`
- CUDA test binary: `atlas_all_cuda_test`
- CUDA test entry point: `tests/cuda/main.cu`
- Some CUDA-incompatible `*.cu` wrappers may be excluded explicitly in the root `CMakeLists.txt`
- Common helpers: `tests/utilities/tests_utils.h`
- `particle_count`-sensitive tests must set the active prefix explicitly instead of assuming it matches capacity
- `System` behavior coverage now lives in `tests/system/system_tests.cpp`
- Source and sink behavior remain covered in `tests/source/` and `tests/sink/`

## File Conventions

- Use `#pragma once` in headers.
- Do not hand-edit generated umbrella headers such as `include/atlas/atlas.h`; use `tools/generate_headers.py`.
- CUDA translation units use `.cu`; CPU translation units use `.cpp`.
- Test sources should include `<testkit/testkit.h>` rather than including GoogleTest headers directly.
- Public APIs are expected to keep Doxygen-style comments where the surrounding file already uses them.

## Dependencies

- **TBB**: required for CPU backend
- **tinyobjloader**: in-tree under `external/tinyobj/`
- **Lyra**: header-only, in-tree under `external/lyra/`
- **OpenGL stack**: needed for vizkit builds (`glfw3`, `GLEW`, `GLU`, `GLUT`)
- **CUDA 12.x**: for GPU builds, with `--expt-relaxed-constexpr --extended-lambda`
