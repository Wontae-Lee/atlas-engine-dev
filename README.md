# Atlas Engine Dev

[![CI](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci.yml/badge.svg)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci.yml)

Atlas is a C++20 particle-simulation engine with a header-only core and dual tasking backends:

- `TBB` for CPU execution
- `CUDA + Thrust` for GPU execution

The core library is exposed through a single umbrella include:

```cpp
#include <atlas/atlas.h>
```

Most of the engine lives under [`include/atlas/`](/home/wontae/CLionProjects/atlas-engine-dev/include/atlas). The only major compiled components are:

- [`src/logging/`](/home/wontae/CLionProjects/atlas-engine-dev/src/logging) for logging
- [`src/vizkit/`](/home/wontae/CLionProjects/atlas-engine-dev/src/vizkit) for optional OpenGL visualization

## What Atlas Provides

- Header-only simulation core with a consistent API across CPU and GPU builds
- Geometry primitives such as box, sphere, cylinder, plane, triangle, and triangle mesh
- Spatial data structures including axis-aligned bounding boxes, spatial hashing, LBVH, and SAH BVH
- Runtime particle emission/removal systems through `Source<T>` and `Sink<T>`
- Per-species velocity generation through `Generator<T>` plus uniform and Maxwell-family generators
- Builder-based construction for most public types
- Portable host/device abstractions such as `DeviceBuffer<T>`, `HostBuffer<T>`, and `device_shared_ptr<T>`
- Optional visualization through Vizkit
- In-tree test coverage for both GoogleTest and CUDA-oriented test flows

## Design Overview

### Header-only core

The engine core is template-heavy and header-only by design. That keeps the simulation layer easy to embed into examples and downstream projects without maintaining a large link-time surface.

The main include tree is under [`include/atlas/`](/home/wontae/CLionProjects/atlas-engine-dev/include/atlas), and [`atlas/atlas.h`](/home/wontae/CLionProjects/atlas-engine-dev/include/atlas/atlas.h) aggregates the public API.

### Exactly one backend

Atlas requires exactly one tasking backend at configure time:

- `ATLAS_USE_TBB=ON`, `ATLAS_USE_CUDA=OFF`
- `ATLAS_USE_TBB=OFF`, `ATLAS_USE_CUDA=ON`

The top-level CMake config rejects:

- both enabled
- both disabled

This keeps backend-dependent aliases stable throughout the codebase.

The runtime particle APIs rely on a simple storage contract:

- `ParticleData<T>` owns the backing buffers
- `ParticleDeviceProbe<T>` exposes raw pointers into those buffers
- `particle_count` is the active prefix length
- `buffer_size` is total allocated capacity

### Backend abstraction

The public API is written against common Atlas types rather than backend-specific containers:

- `DeviceBuffer<T>` maps to `thrust::device_vector<T>` on CUDA and `std::vector<T>` on TBB
- `HostBuffer<T>` maps to `thrust::host_vector<T>` on CUDA and `std::vector<T>` on TBB
- `parallel_for<ExecutionPolicy::host>(...)` dispatches to the active backend-compatible implementation
- `device_shared_ptr<T>` is CUDA managed-memory aware on GPU builds and `std::shared_ptr<T>` on CPU builds

### Builder pattern

Most major types use a fluent nested `Builder` API:

```cpp
const auto domain = system::Domain<float>::builder()
    .with_geometry(box_geometry)
    .with_cell_size(0.02f)
    .make_host_shared();
```

Typical builder endpoints are:

- `.build()` for value construction
- `.make_host_shared()` for shared host ownership

This pattern is used heavily across geometry, simulation systems, fluids, units, codecs, and search structures.

Recent runtime systems follow the same builder style:

```cpp
const auto generator = UniformGenerator<float>::builder()
    .with_min_value(-1.0f)
    .with_max_value(1.0f)
    .with_seed(7u)
    .make_host_shared();

const auto fluid = system::Fluid<float>::builder()
    .add_particle(species_ptr, 1.0f, generator)
    .make_host_shared();
```

## Repository Layout

```text
atlas-engine-dev/
├── include/atlas/                     # Header-only core library
│   ├── atlas.h                        # Umbrella header
│   ├── buffer/                        # HostBuffer, DeviceBuffer
│   ├── codec/                         # Particle codecs / encoding systems
│   ├── container/                     # Fixed-size utility containers
│   ├── core/                          # Portability macros and core helpers
│   ├── data/                          # Particle data/probes
│   ├── domain/                        # Simulation domain
│   ├── generator/                     # Velocity generator interfaces and implementations
│   ├── geometry/                      # Box, Sphere, Plane, Cylinder, Triangle, TriangleMesh
│   ├── logging/                       # Logging public headers
│   ├── math/                          # Vector, matrix, quaternion, reductions
│   ├── matter/                        # Fluid and particle species
│   ├── memory/                        # Shared-pointer and copy abstractions
│   ├── parallel/                      # Backend-agnostic parallel algorithms
│   ├── random/                        # Backend-selected RNG aliases
│   ├── sampling/                      # Sampling helpers
│   ├── searcher/                      # Spatial hashing and neighborhood search
│   ├── sink/                          # Despawn and sink systems
│   ├── source/                        # Spawn/source systems
│   ├── spatial/                       # Ray, AABB, BVH, trace operators
│   ├── sync/                          # Local/world synchronization transforms
│   ├── transform/                     # Transform helpers / reductions
│   ├── tuple/                         # Tuple utilities
│   └── unit/                          # Simulation units
├── src/logging/                       # Compiled logging implementation
├── src/vizkit/                        # Optional OpenGL visualization layer
├── examples/solver/                   # Canonical engine setup example
├── examples/dynamic/                  # Dynamic/Vizkit example
├── src/testkit/                       # Shared testkit headers (GoogleTest shim + cudatest)
├── tests/                             # C++ and CUDA test sources
├── benchmarks/                        # Benchmark targets
├── external/                          # In-tree third-party dependencies
└── tools/                             # Maintenance/codegen helpers
```

## Requirements

### Required for all builds

| Dependency | Notes |
|---|---|
| CMake 3.20+ | Presets are provided |
| C++20 compiler | GCC 11+ or Clang 14+ recommended |
| Ninja or Unix Makefiles | Ninja is the primary preset generator |
| TBB | Required for CPU builds and also linked in CUDA builds |

### Required for CUDA builds

| Dependency | Notes |
|---|---|
| CUDA Toolkit 12.x | `nvcc` is auto-detected from `PATH`, `CUDACXX`, or common install paths |
| NVIDIA driver | Runtime must support the generated toolkit/PTX combination |
| GPU architecture support | Root CMake auto-detects compute capability with `nvidia-smi` and prefers native SASS builds |

### Required for Vizkit

| Dependency | Notes |
|---|---|
| `glfw3` | Windowing |
| `GLEW` | OpenGL extension loading |
| `OpenGL`, `GLU`, `GLUT` | Visualization stack |

### In-tree dependencies

The project vendors several dependencies in [`external/`](/home/wontae/CLionProjects/atlas-engine-dev/external):

- `tinyobj` for OBJ loading
- `lyra` for CLI parsing
- `googletest` for tests
- `googlebenchmark` for benchmarks

## Build Presets

Atlas ships with CMake presets in [CMakePresets.json](/home/wontae/CLionProjects/atlas-engine-dev/CMakePresets.json).

### TBB / CPU presets

| Preset | Build Type | Vizkit | Logging | Tests | Benchmarks |
|---|---|---|---|---|---|
| `tbb-debug` | `Debug` | ON | ON | ON | ON |
| `tbb-release` | `Release` | ON | ON | ON | ON |
| `tbb-relwithdebinfo` | `RelWithDebInfo` | ON | ON | ON | ON |
| `tbb-debug-make` | `Debug` | ON | ON | ON | ON |
| `tbb-debug-headless` | `Debug` | OFF | ON | ON | OFF |

### CUDA / GPU presets

| Preset | Build Type | Vizkit | Logging | Tests | Benchmarks |
|---|---|---|---|---|---|
| `cuda-debug` | `Debug` | ON | ON | OFF | OFF |
| `cuda-release` | `Release` | ON | ON | OFF | OFF |
| `cuda-relwithdebinfo` | `RelWithDebInfo` | ON | ON | OFF | OFF |
| `cuda-debug-headless` | `Debug` | OFF | ON | OFF | OFF |
| `cuda-release-headless` | `Release` | OFF | ON | OFF | OFF |

Current CUDA presets keep tests disabled except for `cuda-debug-tests-headless`, which enables `atlas_all_cuda_test` without Vizkit.

## Continuous Integration

GitHub Actions CI is defined in [`.github/workflows/ci.yml`](/home/wontae/CLionProjects/atlas-engine-dev/.github/workflows/ci.yml).

The current CI job runs a headless Linux TBB build to avoid OpenGL/Vizkit package requirements on hosted runners:

```bash
cmake --preset tbb-debug-headless
cmake --build --preset build-tbb-debug-headless
ctest --preset ctest-tbb-debug-headless
```

The workflow installs:

- `ninja-build`
- `libtbb-dev`

## Quick Start

### 1. Configure and build the recommended headless CPU/TBB build

```bash
cmake --preset tbb-debug-headless
cmake --build --preset build-tbb-debug-headless
```

### 2. Run tests

```bash
ctest --preset ctest-tbb-debug-headless
```

If you want the local Vizkit-enabled developer build instead:

```bash
cmake --preset tbb-debug
cmake --build --preset build-tbb-debug
ctest --preset ctest-tbb-debug
```

### 3. Configure and build a CUDA build

```bash
cmake --preset cuda-debug
cmake --build --preset build-cuda-debug
```

### 4. Headless CUDA build for servers

```bash
cmake --preset cuda-debug-headless
cmake --build --preset build-cuda-debug-headless
```

### 5. CUDA test build

```bash
cmake --preset cuda-debug-tests-headless
cmake --build --preset build-cuda-debug-tests-headless
./build/cuda-debug-tests-headless/atlas_all_cuda_test --gtest_list_tests
```

## Manual CMake Configuration

If you prefer configuring manually instead of using presets:

### CPU / TBB

```bash
cmake -S . -B build/tbb-manual \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DATLAS_USE_TBB=ON \
  -DATLAS_USE_CUDA=OFF \
  -DATLAS_USE_VIZKIT=ON \
  -DATLAS_LOGGING=ON \
  -DATLAS_GOOGLE_TEST=ON \
  -DATLAS_CUDA_TEST=OFF \
  -DATLAS_BENCHMARKS=ON
```

### CUDA

```bash
cmake -S . -B build/cuda-manual \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DATLAS_USE_TBB=OFF \
  -DATLAS_USE_CUDA=ON \
  -DATLAS_USE_VIZKIT=OFF \
  -DATLAS_LOGGING=ON \
  -DATLAS_GOOGLE_TEST=OFF \
  -DATLAS_CUDA_TEST=ON \
  -DATLAS_BENCHMARKS=OFF
```

The root CMake will try to:

- find `nvcc` from `CUDACXX`, `PATH`, `/usr/bin`, or `/usr/local/cuda/bin`
- query `nvidia-smi` for compute capability and driver CUDA runtime support
- choose a native `CMAKE_CUDA_ARCHITECTURES` value when possible
- warn when the CUDA toolkit is newer than the installed driver runtime

You can still override both `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_ARCHITECTURES` explicitly when needed.

## Important CMake Options

| Option | Meaning |
|---|---|
| `ATLAS_USE_TBB` | Enable the TBB backend |
| `ATLAS_USE_CUDA` | Enable the CUDA backend |
| `ATLAS_USE_VIZKIT` | Enable the OpenGL visualization layer |
| `ATLAS_LOGGING` | Enable the logging library and logging macros |
| `ATLAS_GOOGLE_TEST` | Build GoogleTest-based C++ test targets |
| `ATLAS_CUDA_TEST` | Build the aggregated CUDA test executable |
| `ATLAS_BENCHMARKS` | Build benchmark targets |

Notes:

- `ATLAS_USE_TBB` and `ATLAS_USE_CUDA` are mutually exclusive.
- `ATLAS_GOOGLE_TEST` is automatically disabled when `ATLAS_USE_CUDA=ON`.
- Current CUDA presets automatically disable tests and benchmarks.
- The CUDA branch still links TBB because Thrust host-side integration is configured with TBB.

## Usage Example

The following mirrors the setup flow used in [`examples/solver/main.cpp`](/home/wontae/CLionProjects/atlas-engine-dev/examples/solver/main.cpp).

```cpp
#include <atlas/atlas.h>

using namespace atlas;

int main() {
    using sim_t = float;

    const auto box_domain = geometry::Box<sim_t>::builder()
        .with_lower_corner(Vector3<sim_t>{-1.0f, -1.0f, -1.0f})
        .with_upper_corner(Vector3<sim_t>{ 1.0f,  1.0f,  1.0f})
        .make_host_shared();

    const auto domain = system::Domain<sim_t>::builder()
        .with_geometry(box_domain)
        .with_cell_size(0.02f)
        .make_host_shared();

    const auto searcher = system::SpatialHashingSearcher<sim_t>::builder()
        .with_domain(domain)
        .with_range(system::NeighborSearchRange::single)
        .make_host_shared();

    const auto codec = system::SingleCodec<sim_t>::builder()
        .with_domain(domain)
        .make_host_shared();

    const auto nitrogen = FluidicParticle<sim_t>::builder()
        .with_molecular_mass(4.65e-26f)
        .make_host_shared();

    const auto fluid = system::Fluid<sim_t>::builder()
        .add_particle(
            nitrogen,
            1.0f,
            UniformGenerator<sim_t>::builder()
                .with_min_value(-10.0f)
                .with_max_value(10.0f)
                .with_seed(7u)
                .make_host_shared())
        .make_host_shared();

    const auto obstacle_geometry = geometry::Box<sim_t>::builder()
        .with_lower_corner(Vector3<sim_t>{-0.5f, -0.5f, -0.5f})
        .with_upper_corner(Vector3<sim_t>{ 0.5f,  0.5f,  0.5f})
        .make_host_shared();

    const auto sync = system::Sync<sim_t>::builder()
        .make_host_shared();

    const auto unit = system::Unit<sim_t>::builder()
        .with_geometry(obstacle_geometry)
        .with_sync(sync)
        .make_host_shared();

    auto source = system::Source<sim_t>::builder()
        .with_unit(*unit)
        .with_spawn_type(system::SpawnType::Volume)
        .with_spacing(0.05f)
        .with_tolerance(0.0f)
        .with_fluid(fluid)
        .build();

    auto sink = system::Sink<sim_t>::builder()
        .with_unit(*unit)
        .with_despawn_type(system::DespawnType::Surface)
        .with_tolerance(1e-4f)
        .build();

    (void)searcher;
    (void)codec;
    (void)fluid;
    (void)source;
    (void)sink;

    return 0;
}
```

### Typical setup order

In practice, a common Atlas setup sequence is:

1. Define geometry for the world/domain.
2. Build a `Domain<T>` with geometry and cell size.
3. Attach a search structure such as `SpatialHashingSearcher<T>`.
4. Choose a codec for particle representation.
5. Define particle species and assemble fluids, including per-species velocity generators when needed.
6. Create units with geometry plus sync policies.
7. Build `Source<T>` emitters and `Sink<T>` removers around those units.
8. Run host/device simulation steps using the backend selected at configure time.

## Sources, Sinks, and Generators

Atlas includes a runtime pipeline for populating and removing particles from `ParticleDeviceProbe<T>`.

- `Generator<T>` is the abstract interface for velocity generation.
- `UniformGenerator<T>`, `MaxwellSigmaGenerator<T>`, and `MaxwellBoltzmannGenerator<T>` provide concrete policies.
- `Fluid<T>` stores a generator per particle-species entry.
- `Source<T>` uses unit geometry plus a `Fluid<T>` to fill inactive probe slots and initialize species/velocity data.
- `Sink<T>` removes particles matching a geometric despawn condition and compacts the active prefix.

This works best when the probe is treated as an active prefix plus spare capacity:

- initialize `particle_count` to the number of active particles already present
- keep inactive capacity available behind that prefix
- let `Source<T>` reuse inactive slots
- let `Sink<T>` rewrite the active prefix and update `particle_count`

## Geometry and Spatial Features

Atlas currently exposes several geometry and spatial building blocks:

- Analytic geometry:
  - `Box`
  - `Sphere`
  - `Cylinder`
  - `Plane`
  - `Triangle`
- Mesh geometry:
  - `TriangleMesh`
- Spatial utilities:
  - `AxisAlignedBoundingBox`
  - `Ray`
  - `TraceOperator`
  - `QueryOperator`
- Acceleration structures:
  - Spatial hashing
  - Generic BVH interface
  - LBVH
  - SAH BVH

## Examples

### Solver example

Files:

- [`examples/solver/main.cpp`](/home/wontae/CLionProjects/atlas-engine-dev/examples/solver/main.cpp)
- [`examples/solver/main.cu`](/home/wontae/CLionProjects/atlas-engine-dev/examples/solver/main.cu)

This is the best starting point for understanding the engine's object graph and builder flow.

### Dynamic example

Files:

- [`examples/dynamic/main.cpp`](/home/wontae/CLionProjects/atlas-engine-dev/examples/dynamic/main.cpp)
- [`examples/dynamic/main.cu`](/home/wontae/CLionProjects/atlas-engine-dev/examples/dynamic/main.cu)

Use this when exploring the Vizkit/OpenGL integration path.

## Tests

Tests live under [`tests/`](/home/wontae/CLionProjects/atlas-engine-dev/tests) and include both C++ `*.cpp` tests and CUDA `*.cu` test wrappers.

The shared include is:

```cpp
#include <testkit/testkit.h>
```

`testkit` routes to:

- GoogleTest for C++ test builds
- the in-tree `cudatest` implementation for CUDA test builds

Examples of covered areas:

- buffer abstractions
- generators, sources, and sinks
- geometry queries
- axis-aligned bounding boxes and trace operators
- sync operators and transforms
- domains, units, codecs, tuples, iterators, and memory helpers

Recommended GoogleTest/TBB run:

```bash
cmake --preset tbb-debug-headless
cmake --build --preset build-tbb-debug-headless
ctest --preset ctest-tbb-debug-headless
```

CTest presets are also provided:

```bash
ctest --preset ctest-tbb-debug
ctest --preset ctest-tbb-debug-headless
```

CUDA test build:

```bash
cmake --preset cuda-debug-tests-headless
cmake --build --preset build-cuda-debug-tests-headless
./build/cuda-debug-tests-headless/atlas_all_cuda_test --gtest_list_tests
```

Notes:

- CUDA test discovery is done in the root [`CMakeLists.txt`](/home/wontae/CLionProjects/atlas-engine-dev/CMakeLists.txt), not a separate `tests/CMakeLists.txt`.
- `atlas_all_cuda_test` is a single executable driven by [`tests/cuda/main.cu`](/home/wontae/CLionProjects/atlas-engine-dev/tests/cuda/main.cu).
- Some known CUDA-incompatible test wrappers are excluded explicitly from the aggregated CUDA target.

## Benchmarks

Benchmarks are enabled in TBB presets and disabled in CUDA presets. Build them through the normal build step when `ATLAS_BENCHMARKS=ON`.

```bash
cmake --preset tbb-release
cmake --build build/tbb-release -j$(nproc)
```

## Vizkit

Vizkit is optional and only enabled when `ATLAS_USE_VIZKIT=ON`.

It depends on the host OpenGL toolchain:

- `glfw3`
- `GLEW`
- `OpenGL`
- `GLU`
- `GLUT`

If you are building on a headless machine, prefer `tbb-debug-headless`, one of the `cuda-*-headless` presets, or disable Vizkit manually.

## Development Notes

### Generated umbrella headers

Some umbrella headers are generated by tools. In particular:

- [`include/atlas/atlas.h`](/home/wontae/CLionProjects/atlas-engine-dev/include/atlas/atlas.h)
- `vizkit/vizkit.h`

Do not hand-edit generated umbrella headers if they are managed by:

- [`tools/generate_headers.py`](/home/wontae/CLionProjects/atlas-engine-dev/tools/generate_headers.py)

### Header convention

Public headers use:

- `.h` for declarations and documentation
- `.hpp` for inline/template definitions

Typically the `.h` includes the matching `.hpp` at the bottom.

Generated umbrella headers such as [`include/atlas/atlas.h`](/home/wontae/CLionProjects/atlas-engine-dev/include/atlas/atlas.h) should not be edited manually.

### Portability macros

Use Atlas portability macros from [`include/atlas/core/macros.h`](/home/wontae/CLionProjects/atlas-engine-dev/include/atlas/core/macros.h) instead of raw CUDA attributes:

- `ATLAS_HOST`
- `ATLAS_DEVICE`
- `ATLAS_ALL_DEVICE`
- `ATLAS_FORCE_INLINE`

### Logging

Logging is optional at configure time. When enabled, the compiled implementation is provided by [`src/logging/logging.cpp`](/home/wontae/CLionProjects/atlas-engine-dev/src/logging/logging.cpp).

## Known Configuration Constraints

- Benchmarks are not currently enabled for CUDA builds.
- Plane geometry is infinite, so any finite visualization or spawn/sampling behavior must define an explicit finite patch.
- `TriangleMesh` support depends on `tinyobjloader` for OBJ loading.
- Some CUDA test wrappers are still excluded from `atlas_all_cuda_test` due to NVCC/CUDA compatibility issues in specific algorithms or test patterns.
- CUDA configuration attempts to auto-detect `nvcc`, driver runtime support, and `CMAKE_CUDA_ARCHITECTURES`; override those values manually only when detection is unsuitable for your environment.

## License

See the repository's license files and project metadata for the final licensing terms.
