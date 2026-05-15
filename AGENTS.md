# AGENTS.md — Atlas Engine Dev

## Project Overview

Atlas is a C++20 particle simulation engine with a header-only core in `include/atlas/` and an optional OpenGL visualization layer in `src/vizkit/`. Most engine behavior is template-based and inline; compiled sources are primarily in `src/logging/`, `src/vizkit/`, and serialization support under `src/serialization/`.

## Architecture

- **Header-only core**: `include/atlas/` contains the simulation, math, geometry, buffer, source/sink, solver, system, and utility layers. The umbrella header is `include/atlas/atlas.h`.
- **Backend selection**:
  - CMake configuration must enable exactly one of `ATLAS_USE_CUDA` or `ATLAS_USE_TBB`.
  - Compile-time backend code paths use `ATLAS_TASKING_CUDA` or `ATLAS_TASKING_TBB`.
  - `DeviceBuffer<T>` maps to `thrust::device_vector<T>` on CUDA and `std::vector<T>` on TBB.
  - `HostBuffer<T>` maps to `thrust::host_vector<T>` on CUDA and `std::vector<T>` on TBB.
  - `parallel_for<ExecutionPolicy>(...)` dispatches to the active backend implementation.
  - `device_shared_ptr<T>` maps to CUDA-aware managed ownership or `std::shared_ptr<T>`.
- **Core runtime objects**:
  - `atlas::Fluid<T>` owns particle storage, particle/species properties, generator operators, and registered fluid states.
  - `atlas::Universe<T>` owns the Cartesian domain extents, grid resolution, and registered universe states.
  - `atlas::Source<T>` emits particles into inactive fluid capacity.
  - `atlas::Sink<T>` removes particles and compacts the active prefix.
  - `atlas::system::System<T>` is the high-level simulation driver.
- **Simulation orchestration**: `atlas::system::System<T>` stores:
  - one required `FluidHostPtr<T>`
  - one optional `UniverseHostPtr<T>`
  - optional `Source`, `Sink`, `Collider`, and `Orchestrator` subsystems
  - simulation `dt`
  - `update()` runs `source->update(dt) -> orchestrator->update(dt) -> collider->update(dt) or time_integration() -> sink->update(dt)`.
- **Active-prefix particle model**:
  - `Fluid<T>::buffer_size()` is total allocated particle capacity.
  - `Fluid<T>::particle_count()` is the active dense prefix length.
  - `Source<T>` appends into inactive capacity behind that prefix.
  - `Sink<T>` compacts surviving particles back into a dense prefix and updates `particle_count`.
- **Collision / advection**:
  - The old standalone `Advector` API is gone.
  - Collider-aware motion is handled through `Collider<T>` and invoked from `System<T>::advect()`.
  - When no collider is installed, `System<T>::time_integration()` performs `position += velocity * dt`.
- **Orchestration / solver pipeline**:
  - `atlas::system::Orchestrator<T>` coordinates optional search, codec, measurement, field-force/gravity application, and one or more solvers.
  - `System<T>::Builder::with_solver(...)` currently accepts an `OrchestratorHostPtr<T>`.
- **Vizkit**:
  - Vizkit is behind `#ifdef ATLAS_ENABLE_VIZKIT`.
  - `vizkit::Viewer<T>` stores a `SystemHostPtr<T>`.
  - `Viewer<T>` reads `system()->dt()` instead of owning a separate timestep.
- **Namespaces**:
  - Common aliases are re-exported under `atlas::`.
  - High-level runtime types mostly live under `atlas::system::`, `atlas::fluid::`, `atlas::universe::`, and `atlas::geometry::`.
  - Math types live under `atlas::math::` with convenience aliases in `atlas::`.

## Key Patterns

### Builder Pattern

Most public types use nested `Builder` classes with fluent `.with_*()` setters, `.build()`, and `.make_host_shared()`:

```cpp
atlas::HostBuffer<atlas::MatrialProperties<float>> properties(1);
atlas::HostBuffer<atlas::GeneratorHostPtr<float>> generators(1);

properties[0] = atlas::MatrialProperties<float>::builder()
    .with_type(atlas::MaterialType::Molecule)
    .with_molecular_mass(4.651734e-26f)
    .with_reference_diameter(4.17e-10f)
    .build();

generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<float>::builder()
    .with_temperature(300.0f)
    .with_molecular_mass(4.651734e-26f)
    .with_bulk_velocity(atlas::Vector3<float>(0, 0, 0))
    .with_seed(42u)
    .make_host_shared();

const auto fluid = atlas::Fluid<float>::builder()
    .with_buffer_size(4096)
    .with_properties(properties)
    .with_generators(generators)
    .make_host_shared();

const auto sim_system = atlas::system::System<float>::builder()
    .with_fluid(fluid)
    .with_domain(universe)
    .with_source(source)
    .with_sink(sink)
    .with_collider(collider)
    .with_dt(0.01f)
    .make_host_shared();
```

Notes:
- `Source<T>::Builder` uses `with_units(...)`, not `with_unit(...)`.
- `Sink<T>::Builder` uses `with_units(...)`, not `with_unit(...)`.
- `System<T>::Builder` does not own buffer sizing; particle capacity belongs to `Fluid<T>`.

### Active Prefix Discipline

When writing code or tests that touch fluid particle state:

1. set `buffer_size` through `Fluid<T>::Builder`
2. explicitly set `particle_count` when active particles should exist
3. treat `[0, particle_count)` as the only active particle range
4. leave the tail available for source emission or later compaction

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

- `buffer`, `codec`, `collider`, `container`, `core`
- `fluid`, `generator`, `geometry`, `indexer`, `iterator`
- `logging`, `material`, `math`, `measure`, `memory`, `observer`, `orchestrator`
- `parallel`, `random`, `remove`, `sampling`, `scan`, `searcher`, `serialization`, `shuffle`
- `sink`, `solver`, `source`, `spatial`, `sync`, `system`
- `transform`, `tuple`, `unit`, `universe`

There is no `include/atlas/advector/` module.

## Vizkit Notes

- `src/vizkit/viewer/viewer.h` depends on `atlas/system/system.h`.
- `Viewer<T>::Builder` uses `.with_system(...)`, `.with_size(...)`, `.with_title(...)`, and `.with_fullscreen(...)`.
- Layer updates still receive `(GLFWwindow*, Camera&, T dt)`, but that `dt` comes from the bound `System`.
- Vizkit code is compiled only when `ATLAS_ENABLE_VIZKIT` is defined.

## Build & Test

`CMakePresets.json` requires CMake 3.20+. All presets use the Ninja generator.

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

Available configure presets from `CMakePresets.json`:

- TBB: `tbb-debug`, `tbb-release`
- CUDA: `cuda-debug`, `cuda-release`, `cuda-debug-tests`

Available build presets:

- `build-tbb-debug`, `build-tbb-release`
- `build-cuda-debug`, `build-cuda-release`, `build-cuda-debug-tests`

Available test presets:

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

- exactly one of `ATLAS_USE_CUDA` / `ATLAS_USE_TBB` must be enabled
- `ATLAS_GOOGLE_TEST` is disabled for CUDA presets
- benchmarks are currently TBB-only

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
- Common helpers: `tests/utilities/test_utils.h`
- `particle_count`-sensitive tests must set the active prefix explicitly instead of assuming it matches capacity
- `System` behavior coverage lives in `tests/system/system_tests.cpp`
- Source and sink behavior remain covered in `tests/source/` and `tests/sink/`
- Observer behavior is covered under `tests/observer/`
- Material-property behavior is covered under `tests/material/`

## File Conventions

- Use `#pragma once` in headers.
- Do not hand-edit generated umbrella headers such as `include/atlas/atlas.h`; use `tools/generate_headers.py`.
- CUDA translation units use `.cu`; CPU translation units use `.cpp`.
- Test sources should include `<testkit/testkit.h>` rather than including GoogleTest headers directly.
- Public APIs are expected to keep Doxygen-style comments where the surrounding file already uses them.
- The project currently uses the public alias `MatrialProperties` for material records. Keep naming consistent with existing code unless the task explicitly includes a rename.
- Observer APIs and CSV outputs currently use the `sensor_matrics` naming found in the codebase. Do not silently normalize that spelling in isolated edits.

## Dependencies

External:

- **TBB**: required for the TBB backend
- **CUDA 12.x**: required for the CUDA backend, with `--expt-relaxed-constexpr --extended-lambda`
- **OpenGL stack**: required for Vizkit builds (`glfw3`, `GLEW`, `GLU`, `GLUT`)

In-tree under `external/`:

- **tinyobjloader** (`external/tinyobj/`) — OBJ mesh loading
- **Lyra** (`external/lyra/`) — header-only CLI argument parsing
- **googletest** (`external/googletest/`) — C++ unit test framework
- **googlebenchmark** (`external/googlebenchmark/`) — microbenchmark framework
- **protobuf** (`external/protobuf/`) — binary snapshot serialization
