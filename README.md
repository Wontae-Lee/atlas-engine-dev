<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine Dev

Atlas is a C++20 particle simulation engine with a header-only core, selectable TBB or CUDA execution backends, optional serialization support, and an OpenGL visualization layer called Vizkit.

The public API is aggregated through:

```cpp
#include <atlas/atlas.h>
```

## Overview

Atlas is built around a small set of runtime objects:

- `atlas::Fluid<T>` owns particle storage, material properties, particle generators, observers, and registered fluid states
- `atlas::Universe<T>` owns the Cartesian simulation domain, grid resolution, observers, and registered universe states
- `atlas::Source<T>` emits particles into inactive capacity behind the fluid active prefix
- `atlas::Sink<T>` removes particles and compacts survivors back into a dense active prefix
- `atlas::system::Orchestrator<T>` coordinates search, codec, measurement, field forces, gravity, and solver stages
- `atlas::system::System<T>` is the top-level step driver

`System<T>::update()` runs the current simulation step in this order:

1. `source->update(dt)`
2. `orchestrator->update(dt)`
3. `collider->update(dt)` when a collider is installed, otherwise fallback time integration
4. `sink->update(dt)`

When no collider is installed, fallback motion is:

```cpp
position += velocity * dt;
```

## Backend Model

Exactly one backend must be active at configure time:

- `ATLAS_USE_TBB=ON`, `ATLAS_USE_CUDA=OFF`
- `ATLAS_USE_TBB=OFF`, `ATLAS_USE_CUDA=ON`

Atlas keeps most runtime APIs backend-agnostic through shared aliases:

- `DeviceBuffer<T>` maps to `std::vector<T>` on TBB and `thrust::device_vector<T>` on CUDA
- `HostBuffer<T>` maps to `std::vector<T>` on TBB and `thrust::host_vector<T>` on CUDA
- `device_shared_ptr<T>` maps to `std::shared_ptr<T>` on TBB and CUDA-aware managed ownership on CUDA
- `parallel_for<ExecutionPolicy>(...)` dispatches to the active backend implementation

Backend-specific compile-time paths use `ATLAS_TASKING_TBB` or `ATLAS_TASKING_CUDA`.

## Key Concepts

### Active Prefix Storage

Particle storage separates allocated capacity from the logical particle population:

- `Fluid<T>::buffer_size()` is total allocated capacity
- `Fluid<T>::particle_count()` is the active dense prefix length
- active particles live in `[0, particle_count)`
- inactive capacity begins at `particle_count`

This contract matters for source and sink behavior:

- `Source<T>` appends into inactive capacity without assuming capacity equals population
- `Sink<T>` compacts surviving particles back into `[0, particle_count)`
- tests should set `particle_count` explicitly when active particles should exist

### Builder Pattern

Most public types use nested builders with fluent `.with_*()` setters and `.build()` / `.make_host_shared()` endpoints.

```cpp
const auto fluid = atlas::Fluid<float>::builder()
    .with_buffer_size(4096)
    .with_properties(properties)
    .with_generators(generators)
    .make_host_shared();

const auto system = atlas::System<float>::builder()
    .with_fluid(fluid)
    .with_domain(universe)
    .with_source(source)
    .with_sink(sink)
    .with_collider(collider)
    .with_solver(orchestrator)
    .with_dt(0.01f)
    .make_host_shared();
```

Notes:

- `Source<T>::Builder` and `Sink<T>::Builder` use `.with_units(...)`
- `System<T>::Builder::with_solver(...)` accepts an `OrchestratorHostPtr<T>`
- particle capacity belongs to `Fluid<T>`, not `System<T>`

### Namespaces

- common aliases are re-exported under `atlas::`
- simulation orchestration lives under `atlas::system::`
- fluid source/sink implementation types live under `atlas::fluid::`
- geometry lives under `atlas::geometry::`
- math lives under `atlas::math::`, with common vector and matrix aliases re-exported in `atlas::`

## Repository Layout

```text
atlas-engine-dev/
├── include/atlas/            # Header-only core library
│   ├── atlas.h               # Generated umbrella header
│   ├── buffer/
│   ├── codec/
│   ├── collider/
│   ├── container/
│   ├── core/
│   ├── fluid/
│   ├── generator/
│   ├── geometry/
│   ├── indexer/
│   ├── iterator/
│   ├── logging/
│   ├── material/
│   ├── math/
│   ├── measure/
│   ├── memory/
│   ├── observer/
│   ├── orchestrator/
│   ├── parallel/
│   ├── random/
│   ├── remove/
│   ├── sampling/
│   ├── scan/
│   ├── searcher/
│   ├── serialization/
│   ├── shuffle/
│   ├── sink/
│   ├── solver/
│   ├── source/
│   ├── spatial/
│   ├── sync/
│   ├── system/
│   ├── transform/
│   ├── tuple/
│   ├── unit/
│   └── universe/
├── src/logging/              # Compiled logging implementation
├── src/serialization/        # Snapshot serialization support
├── src/testkit/              # Shared C++ and CUDA test shim
├── src/vizkit/               # Optional OpenGL visualization layer
├── assets/                   # Mesh assets used by examples (OBJ/MTL)
├── docs/                     # Logo, Doxygen, and developer metadata
├── examples/                 # cylinder, honeycomb, inflow, intake, orchestrator, waterfall
├── tests/                    # GoogleTest and CUDA test sources
├── benchmarks/
├── external/
└── tools/
```

## Features

- header-only simulation core
- TBB and CUDA/Thrust execution backends
- active-prefix particle storage for source emission and sink compaction
- material records and Maxwell/thermal particle generators
- analytic geometry including box, sphere, cylinder, plane, circle, square, triangle, and triangle mesh
- spatial primitives and acceleration structures including rays, AABBs, spatial hashing, BVH, LBVH, and SAH BVH
- source, sink, collider, orchestrator, observer, and serialization runtime modules
- DSMC and SPH solver modules
- optional Vizkit viewer and layer system

## Requirements

### Core

| Dependency | Notes |
|---|---|
| CMake 3.20+ | Presets are provided in [`CMakePresets.json`](CMakePresets.json) |
| C++20 compiler | GCC 11+ and Clang 14+ are reasonable targets |
| Ninja | All presets use Ninja |
| TBB | Required for TBB builds |

### CUDA

| Dependency | Notes |
|---|---|
| CUDA Toolkit 12.x | Required for CUDA builds |
| NVIDIA driver | Must support the selected toolkit/runtime combination |

### Vizkit

| Dependency | Notes |
|---|---|
| `glfw3` | Windowing |
| `GLEW` | Extension loading |
| `OpenGL` / `GLU` / `GLUT` | Rendering stack |

### In-tree Dependencies

All vendored under [`external/`](external/):

- `tinyobjloader` — OBJ mesh loading
- `lyra` — CLI argument parsing
- `googletest` — C++ unit test framework
- `googlebenchmark` — microbenchmark framework
- `protobuf` — binary snapshot serialization

## Build Presets

### Configure Presets

TBB:

- `tbb-debug`
- `tbb-release`

CUDA:

- `cuda-debug`
- `cuda-release`
- `cuda-debug-tests`

### Build Presets

- `build-tbb-debug`
- `build-tbb-release`
- `build-cuda-debug`
- `build-cuda-release`
- `build-cuda-debug-tests`

### Test Presets

- `ctest-tbb-debug`
- `ctest-tbb-release`

## Quick Start

### Linux

```bash
sudo apt-get update
sudo apt-get install -y ninja-build libtbb-dev
sudo apt-get install -y libglfw3-dev libglew-dev freeglut3-dev
```

Debug build:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

### macOS

```bash
brew update
brew install ninja tbb
brew install glfw glew freeglut
```

Debug build:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(sysctl -n hw.ncpu)
ctest --preset ctest-tbb-debug
```

### CUDA

Debug build:

```bash
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)
```

CUDA test build:

```bash
cmake --preset cuda-debug-tests
cmake --build build/cuda-debug-tests --target atlas_all_cuda_test -j$(nproc)
./build/cuda-debug-tests/atlas_all_cuda_test --gtest_list_tests
```

## Important CMake Options

| Option | Meaning |
|---|---|
| `ATLAS_USE_TBB` | Enable the TBB backend |
| `ATLAS_USE_CUDA` | Enable the CUDA backend |
| `ATLAS_USE_VIZKIT` | Enable Vizkit |
| `ATLAS_LOGGING` | Build logging support |
| `ATLAS_GOOGLE_TEST` | Build GoogleTest-based C++ tests |
| `ATLAS_CUDA_TEST` | Build the aggregated CUDA test executable |
| `ATLAS_BENCHMARKS` | Build benchmarks |

Constraints:

- `ATLAS_USE_TBB` and `ATLAS_USE_CUDA` are mutually exclusive and exactly one must be enabled
- CUDA presets disable GoogleTest-based C++ tests
- benchmarks are currently TBB-only

## Minimal Runtime Example

The current runtime model is `Fluid + Universe + optional subsystems + System`.

```cpp
#include <atlas/atlas.h>

int main() {
    using T = float;

    const auto observer = atlas::Observer::builder()
        .with_source_sensor_metrics(1024)
        .with_sink_sensor_metrics(1024)
        .make_host_shared();

    atlas::HostBuffer<atlas::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0] = atlas::MaterialProperties<T>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_molecular_mass(4.651734e-26f)
        .with_reference_diameter(4.17e-10f)
        .build();

    generators[0] = atlas::fluid::MaxwellBoltzmannGenerator<T>::builder()
        .with_temperature(300.0f)
        .with_molecular_mass(4.651734e-26f)
        .with_bulk_velocity(atlas::Vector3<T>(0, 0, 0))
        .with_seed(42u)
        .make_host_shared();

    const auto fluid = atlas::Fluid<T>::builder()
        .with_buffer_size(4096)
        .with_properties(properties)
        .with_generators(generators)
        .with_observer(observer)
        .make_host_shared();

    const auto universe = atlas::Universe<T>::builder()
        .with_lower_corner(atlas::Vector3<T>(-1, -1, -1))
        .with_upper_corner(atlas::Vector3<T>(1, 1, 1))
        .with_cell_size(0.1f)
        .with_observer(observer)
        .make_host_shared();

    const auto source_geometry = atlas::geometry::Box<T>::builder()
        .with_lower_corner(atlas::Vector3<T>(-0.8f, -0.2f, -0.2f))
        .with_upper_corner(atlas::Vector3<T>(-0.4f, 0.2f, 0.2f))
        .make_host_shared();

    atlas::HostBuffer<atlas::Unit<T>> source_units(1);
    source_units[0] = atlas::Unit<T>::builder()
        .with_geometry(source_geometry)
        .with_sync(atlas::Sync<T>::builder().make_host_shared())
        .build();

    atlas::HostBuffer<atlas::fluid::SpawnType> spawn_types(1);
    spawn_types[0] = atlas::fluid::SpawnType::Volume;

    const auto source = atlas::Source<T>::builder()
        .with_units(source_units)
        .with_fluid(fluid)
        .with_observer(observer)
        .with_spawn_types(spawn_types)
        .with_spawn_operator(atlas::fluid::SpawnOperator<T>(atlas::fluid::SpawnType::Volume))
        .with_spacing(0.05f)
        .make_host_shared();

    const auto system = atlas::System<T>::builder()
        .with_fluid(fluid)
        .with_domain(universe)
        .with_source(source)
        .with_dt(0.01f)
        .make_host_shared();

    system->update();
    return 0;
}
```

For a complete end-to-end setup with DSMC, collider interaction, sink removal, observer export, and optional Vizkit rendering, see [`examples/cylinder/main.cpp`](examples/cylinder/main.cpp).

## Tests

Tests live under [`tests/`](tests/) and include [`src/testkit/testkit.h`](src/testkit/testkit.h) through:

```cpp
#include <testkit/testkit.h>
```

Current test structure:

- C++ tests are discovered recursively from `tests/*.cpp`
- CUDA tests are discovered recursively from `tests/*.cu`
- aggregate C++ binary: `atlas_tests`
- per-directory C++ binaries: `atlas_tests_<directory>`
- aggregate CUDA binary: `atlas_all_cuda_test`
- CUDA test entry point: [`tests/cuda/main.cu`](tests/cuda/main.cu)

Recommended TBB test run:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

Recommended CUDA test run:

```bash
cmake --preset cuda-debug-tests
cmake --build build/cuda-debug-tests --target atlas_all_cuda_test -j$(nproc)
./build/cuda-debug-tests/atlas_all_cuda_test --gtest_list_tests
```

## Examples

Maintained examples live under [`examples/`](examples/):

- [`examples/cylinder/`](examples/cylinder/) demonstrates DSMC cylinder flow, collider interaction, sink removal, observer export, and optional Vizkit rendering
- [`examples/honeycomb/`](examples/honeycomb/) demonstrates honeycomb channel flow with mesh-based geometry
- [`examples/inflow/`](examples/inflow/) demonstrates source-driven particle inflow
- [`examples/intake/`](examples/intake/) demonstrates intake-style source and sink setup
- [`examples/orchestrator/`](examples/orchestrator/) demonstrates orchestrator-centered simulation wiring
- [`examples/waterfall/`](examples/waterfall/) demonstrates a waterfall-style particle scenario

Each example provides `main.cpp` and `main.cu` entry points selected by the active backend.

## Development Notes

- use portability macros from [`include/atlas/core/macros.h`](include/atlas/core/macros.h): `ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE`, `ATLAS_FORCE_INLINE`, `ATLAS_NODISCARD`, `ATLAS_MAYBE_UNUSED`, and `RESTRICT`
- template modules follow paired `.h` and `.hpp` files, with the `.h` including the `.hpp` at the bottom
- do not hand-edit generated umbrella headers such as [`include/atlas/atlas.h`](include/atlas/atlas.h); use [`tools/generate_headers.py`](tools/generate_headers.py)
- a public rename is a breaking change; rename public spellings only when explicitly requested
- Vizkit is compiled only when enabled and reads `system()->dt()` from the bound `System`

## License

See the repository license files and project metadata for licensing terms.
