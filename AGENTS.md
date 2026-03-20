# AGENTS.md — Atlas Engine Dev

## Project Overview

Atlas is a C++20 header-only particle simulation engine with dual CUDA/TBB backends. The core library (`atlas::core`) lives entirely in `include/atlas/` — only `src/logging/` (compiled static lib) and `src/vizkit/` (OpenGL visualization) contain `.cpp` files.

## Architecture

- **Header-only core**: `include/atlas/` — all templates, no link step. The umbrella header is `atlas/atlas.h`.
- **Backend abstraction**: A single `ATLAS_TASKING_CUDA` or `ATLAS_TASKING_TBB` define selects between Thrust (GPU) and TBB (CPU). Exactly one must be active. Key abstractions:
  - `DeviceBuffer<T>` → `thrust::device_vector<T>` or `std::vector<T>`
  - `HostBuffer<T>` → `thrust::host_vector<T>` or `std::vector<T>`
  - `parallel_for<ExecutionPolicy>(...)` → Thrust or TBB dispatch
  - `device_shared_ptr<T>` → CUDA managed-memory ref-counted ptr or `std::shared_ptr<T>`
- **Vizkit** (`src/vizkit/`): Optional OpenGL visualization layer, guarded by `#ifdef ATLAS_ENABLE_VIZKIT`.
- **Namespaces**: Core types use `atlas::` (e.g., `Vector3`, `DeviceBuffer`). Simulation systems use `atlas::system::` (Domain, Codec, Searcher, Fluid, Generator, Source, Sink, Unit, Sync, System). Geometry uses `atlas::geometry::`. Math lives in `atlas::math::` with convenience aliases in `atlas::`.
- **Particle ownership split**:
  - `ParticleData<T>` owns storage buffers and exposes one `ParticleDeviceProbe<T>`
  - `ParticleDeviceProbe<T>::particle_count` is the active prefix length, not merely the total capacity
  - `ParticleDeviceProbe<T>::buffer_size` is the actual allocated capacity
- **Emission / removal pipeline**:
  - `Source<T>` fills inactive slots (`active == 0`) inside a `ParticleDeviceProbe<T>`
  - `Sink<T>` compacts the active prefix in-place via `remove_if`
  - `Fluid<T>` now carries per-species `amounts` and optional velocity `generators`

## Key Patterns

### Builder Pattern (Pervasive)
Every major type uses a nested `Builder` with fluent `.with_*()` setters, `.build()` for value construction, and `.make_host_shared()` for `host_shared_ptr` ownership:
```cpp
const auto domain = system::Domain<sim_t>::builder()
    .with_geometry(box)
    .with_cell_size(0.02f)
    .make_host_shared();
```
See `examples/solver/main.cpp` for the canonical usage flow.

Recent additions follow the same pattern:
```cpp
const auto generator = UniformGenerator<float>::builder()
    .with_min_value(-1.0f)
    .with_max_value(1.0f)
    .with_seed(7u)
    .make_host_shared();

const auto fluid = system::Fluid<float>::builder()
    .add_particle(species_ptr, 1.0f, generator)
    .make_host_shared();

const auto source = system::Source<float>::builder()
    .with_unit(unit)
    .with_spawn_type(system::SpawnType::Volume)
    .with_spacing(0.02f)
    .with_fluid(fluid)
    .build();
```

### Device Portability Macros
All host/device-annotated functions use macros from `include/atlas/core/macros.h`:
- `ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE` — expand to CUDA qualifiers or no-ops
- `ATLAS_FORCE_INLINE` — compiler-specific force-inline
- `ATLAS_NODISCARD`, `ATLAS_MAYBE_UNUSED`, `RESTRICT`

Always use these macros instead of raw `__host__`/`__device__`.

### Header / Implementation Split
Templates use a `.h` (declarations + docs) / `.hpp` (definitions) pair in the same directory. The `.h` includes the `.hpp` at the bottom. Example: `geometry/box.h` includes `geometry/box.hpp`.

## Build & Test

```bash
# Configure + build (TBB/CPU, Debug, with tests)
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)

# Run tests
ctest --test-dir build/tbb-debug --output-on-failure

# CUDA build (tests/benchmarks auto-disabled)
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)
```
Key CMake options: `ATLAS_USE_CUDA`, `ATLAS_USE_TBB` (mutually exclusive), `ATLAS_USE_VIZKIT`, `ATLAS_LOGGING`, `ATLAS_TESTS`, `ATLAS_BENCHMARKS`. See `CMakePresets.json` for all presets.

## Tests

- Framework: GoogleTest (in-tree at `external/googletest/`)
- All test sources: `tests/<module>/*_tests.cpp` — auto-discovered via `GLOB_RECURSE`
- Single test binary: `atlas_tests` (links `atlas::core` + GTest)
- Test helpers: `tests/utilities/tests_utils.h` — provides `test::near()`, `test::vec_near()`, `test::is_finite_vec()`, `test::all_finite_points()`, `test::points_in_range()`, `test::point_buffers_near()`
- Tests include `#include "../utilities/tests_utils.h"` then `#include <atlas/atlas.h>`
- Generator coverage lives under `tests/generator/`; sink/source behavior lives under `tests/sink/` and `tests/source/`

## File Conventions

- All headers use `#pragma once` (no include guards). The tool `tools/update_header_guard.py` enforces this.
- `tools/generate_headers.py` auto-generates umbrella headers (`atlas/atlas.h`, `vizkit/vizkit.h`) — do not hand-edit these files.
- CUDA source files use `.cu` extension; CPU equivalents use `.cpp`. Examples provide both (`main.cpp` + `main.cu`).
- Doxygen-style `/** */` comments on all public APIs with `@brief`, `@details`, `@tparam`, `@note`, `@warning`.
- `particle_count`-sensitive tests should set the active prefix explicitly rather than assuming it equals buffer capacity.

## Dependencies

- **TBB** (system): `find_package(TBB REQUIRED)` — must be installed
- **tinyobjloader** (in-tree): `external/tinyobj/` — OBJ mesh loading
- **Lyra** (in-tree, header-only): `external/lyra/` — CLI argument parsing
- **OpenGL stack** (vizkit only): glfw3, GLEW, GLU, GLUT — system packages
- **CUDA 12.x** (GPU builds): arch 86 default, nvcc flags `--expt-relaxed-constexpr --extended-lambda`
