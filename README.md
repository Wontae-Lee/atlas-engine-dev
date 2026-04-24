# Atlas Engine Dev

[![TBB Core Linux](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-core-linux.yml/badge.svg)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-core-linux.yml)
[![TBB Vizkit Linux](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-vizkit-linux.yml/badge.svg)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-vizkit-linux.yml)
[![TBB Core macOS](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-core-macos.yml/badge.svg)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-core-macos.yml)
[![TBB Vizkit macOS](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-vizkit-macos.yml/badge.svg)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb-vizkit-macos.yml)

Atlas is a C++20 particle simulation engine with:

- a header-only core in [`include/atlas/`](include/atlas/)
- a TBB backend for CPU execution
- a CUDA + Thrust backend for GPU execution
- an optional OpenGL visualization layer in [`src/vizkit/`](src/vizkit/)

The public API is aggregated through:

```cpp
#include <atlas/atlas.h>
```

## Overview

Atlas is built around a small set of runtime objects:

- `atlas::Fluid<T>` owns particle storage, species properties, generators, and registered particle states
- `atlas::Universe<T>` owns the Cartesian simulation domain and registered cell states
- `atlas::Source<T>` emits particles into inactive fluid capacity
- `atlas::Sink<T>` removes particles and compacts the active prefix
- `atlas::system::System<T>` coordinates one simulation step

The current high-level simulation entry point is `atlas::system::System<T>`. Its default update order is:

1. `emit()`
2. `orchestrate()`
3. `advect()`
4. `remove()`

If no collider is installed, `System<T>::advect()` falls back to direct time integration.

## Backend Model

Exactly one backend must be active at configure time:

- `ATLAS_USE_TBB=ON`, `ATLAS_USE_CUDA=OFF`
- `ATLAS_USE_TBB=OFF`, `ATLAS_USE_CUDA=ON`

Atlas keeps most runtime APIs backend-agnostic through shared aliases:

- `DeviceBuffer<T>` maps to `std::vector<T>` on TBB and `thrust::device_vector<T>` on CUDA
- `HostBuffer<T>` maps to `std::vector<T>` on TBB and `thrust::host_vector<T>` on CUDA
- `device_shared_ptr<T>` maps to `std::shared_ptr<T>` on TBB and managed-memory ownership on CUDA
- `parallel_for(...)` dispatches to the active backend implementation

## Key Concepts

### Active Prefix Storage

Particle storage distinguishes logical population from allocated capacity:

- `buffer_size` is total allocated capacity
- `particle_count` is the active dense prefix length

This contract is important throughout the runtime:

- `Source<T>` writes into inactive capacity behind the active prefix
- `Sink<T>` compacts surviving particles back into a dense prefix
- tests should set `particle_count` explicitly instead of assuming it matches capacity

### Builder Pattern

Most public types use nested builders with fluent setters and `.build()` / `.make_host_shared()` endpoints.

```cpp
const auto fluid = atlas::Fluid<float>::builder()
    .with_buffer_size(4096)
    .with_properties(properties)
    .with_generators(generators)
    .make_host_shared();

const auto system = atlas::system::System<float>::builder()
    .with_fluid(fluid)
    .with_domain(universe)
    .with_source(source)
    .with_sink(sink)
    .with_dt(0.01f)
    .make_host_shared();
```

### Namespaces

- core aliases are available under `atlas::`
- high-level runtime types are implemented under `atlas::system::`
- geometry lives under `atlas::geometry::`
- math lives under `atlas::math::` with common aliases re-exported in `atlas::`

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
│   ├── data/
│   ├── domain/
│   ├── flatten/
│   ├── generator/
│   ├── geometry/
│   ├── indexer/
│   ├── iterator/
│   ├── logging/
│   ├── math/
│   ├── material/
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
├── src/vizkit/               # Optional OpenGL visualization layer
├── src/testkit/              # Shared test shim for C++ and CUDA tests
├── examples/cylinder/        # DSMC cylinder example
├── tests/                    # C++ and CUDA tests
├── benchmarks/
├── external/
└── tools/
```

## Features

- header-only simulation core
- portable particle storage and execution abstractions across TBB and CUDA
- analytic geometry including box, sphere, cylinder, plane, circle, square, triangle, and triangle mesh
- spatial acceleration structures including spatial hashing, BVH, LBVH, and SAH BVH
- source/sink runtime pipeline for particle emission and removal
- DSMC and SPH solver modules
- observer and serialization support
- optional Vizkit viewer and layer system

## Requirements

### Core

| Dependency | Notes |
|---|---|
| CMake 3.20+ | Presets are provided in [`CMakePresets.json`](CMakePresets.json) |
| C++20 compiler | GCC 11+ / Clang 14+ class toolchains are reasonable targets |
| Ninja or Unix Makefiles | Most presets use Ninja |
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

- `tinyobjloader` in `external/tinyobj/`
- `lyra` in `external/lyra/`
- GoogleTest and benchmark dependencies are also vendored for enabled builds

## Build Presets

### TBB

- `tbb-debug`
- `tbb-release`
- `tbb-debug-core`
- `tbb-release-core`
- `tbb-relwithdebinfo`
- `tbb-debug-make`

### CUDA

- `cuda-debug`
- `cuda-release`
- `cuda-debug-core`
- `cuda-release-core`
- `cuda-debug-tests`
- `cuda-relwithdebinfo`

### Build Presets

- `build-tbb-debug`
- `build-tbb-release`
- `build-tbb-debug-core`
- `build-tbb-release-core`
- `build-tbb-relwithdebinfo`
- `build-cuda-debug`
- `build-cuda-release`
- `build-cuda-debug-core`
- `build-cuda-release-core`
- `build-cuda-debug-tests`
- `build-cuda-relwithdebinfo`

### Test Presets

- `ctest-tbb-debug`
- `ctest-tbb-release`
- `ctest-tbb-debug-core`
- `ctest-tbb-release-core`

## Quick Start

### Linux

```bash
sudo apt-get update
sudo apt-get install -y ninja-build libtbb-dev
sudo apt-get install -y libglfw3-dev libglew-dev
```

Recommended core build:

```bash
cmake --preset tbb-debug-core
cmake --build build/tbb-debug-core -j$(nproc)
ctest --preset ctest-tbb-debug-core
```

Vizkit-enabled local build:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

### macOS

```bash
brew update
brew install ninja tbb
brew install glfw glew
```

Recommended core build:

```bash
cmake --preset tbb-debug-core
cmake --build build/tbb-debug-core -j$(sysctl -n hw.ncpu)
ctest --preset ctest-tbb-debug-core
```

### CUDA

Debug build:

```bash
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)
```

Core CUDA build:

```bash
cmake --preset cuda-debug-core
cmake --build build/cuda-debug-core -j$(nproc)
```

CUDA test build:

```bash
cmake --preset cuda-debug-tests
cmake --build build/cuda-debug-tests --target atlas_all_cuda_test -j$(nproc)
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

- `ATLAS_USE_TBB` and `ATLAS_USE_CUDA` are mutually exclusive
- CUDA presets disable GoogleTest-based C++ tests
- benchmarks are currently enabled only in TBB-oriented presets

## Minimal Runtime Example

The current runtime model is `Fluid + Universe + optional subsystems + System`.

```cpp
#include <atlas/atlas.h>

int main() {
    using T = float;

    const auto observer = atlas::Observer::builder()
        .with_source_sensor_matrics(1024)
        .with_sink_sensor_matrics(1024)
        .make_host_shared();

    atlas::HostBuffer<atlas::MatrialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0] = atlas::MatrialProperties<T>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_molecular_mass(4.651734e-26f)
        .with_collision_diameter(4.17e-10f)
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

    const auto system = atlas::system::System<T>::builder()
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

Tests live under [`tests/`](tests/) and use:

```cpp
#include <testkit/testkit.h>
```

Current test structure:

- C++ tests: recursive `tests/*.cpp`
- CUDA tests: recursive `tests/*.cu`
- aggregate C++ binary: `atlas_tests`
- per-directory C++ binaries: `atlas_tests_<directory>`
- aggregate CUDA binary: `atlas_all_cuda_test`

Recommended runs:

```bash
cmake --preset tbb-debug-core
cmake --build build/tbb-debug-core -j$(nproc)
ctest --preset ctest-tbb-debug-core
```

```bash
cmake --preset cuda-debug-tests
cmake --build build/cuda-debug-tests --target atlas_all_cuda_test -j$(nproc)
./build/cuda-debug-tests/atlas_all_cuda_test --gtest_list_tests
```

## Example

The main maintained example is the cylinder flow case:

- [`examples/cylinder/main.cpp`](examples/cylinder/main.cpp)
- [`examples/cylinder/main.cu`](examples/cylinder/main.cu)

It demonstrates:

- thermal nitrogen particle emission
- DSMC hard-sphere collisions
- collider interaction against a central cylinder
- sink removal outside the domain
- observer export
- optional Vizkit visualization

## Development Notes

- use portability macros from [`include/atlas/core/macros.h`](include/atlas/core/macros.h): `ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE`, `ATLAS_FORCE_INLINE`
- template modules follow paired `.h` and `.hpp` files, with the `.h` including the `.hpp` at the bottom
- do not hand-edit generated umbrella headers such as [`include/atlas/atlas.h`](include/atlas/atlas.h); use [`tools/generate_headers.py`](tools/generate_headers.py)
- Vizkit depends on `atlas/system/system.h` and reads `system()->dt()` from the bound system

## License

See the repository license files and project metadata for licensing terms.
