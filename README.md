# atlas-core-dev

C++20 header-only particle simulation engine with dual **CUDA** / **TBB** backends.

## Features

- **Header-only core** — single `#include <atlas/atlas.h>` pulls in the entire library
- **Dual backend** — compile-time selection between GPU (CUDA/Thrust) and CPU (TBB)
- **Geometry primitives** — Box, Sphere, Cylinder, Plane, Triangle, TriangleMesh
- **Spatial acceleration** — Spatial hashing, BVH, LBVH, SAH-BVH
- **Fluent builder API** — every major type uses `.builder().with_*().make_host_shared()`
- **Vizkit** — optional OpenGL real-time visualization layer
- **Logging** — leveled logging with debug-only check macros

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| CMake | ≥ 3.20 | Ninja recommended |
| C++ Compiler | C++20 support | GCC 11+, Clang 14+ |
| TBB | system | `libtbb-dev` (Ubuntu) |
| CUDA Toolkit | 12.x | Required for GPU builds, arch 86 default |
| OpenGL stack | glfw3, GLEW, GLU, GLUT | Required when Vizkit is enabled |

## Quick Start

### TBB (CPU) Build

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
```

### CUDA (GPU) Build

```bash
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)
```

### Docker Build

```bash
docker build -t atlas .
# CPU-only
docker build --build-arg ATLAS_USE_CUDA=OFF -t atlas-cpu .
```

## CMake Options

| Option | Default | Description |
|---|---|---|
| `ATLAS_USE_TBB` | `ON` | Enable TBB (CPU) backend |
| `ATLAS_USE_CUDA` | `OFF` | Enable CUDA (GPU) backend |
| `ATLAS_USE_VIZKIT` | `ON` | Build OpenGL visualization layer |
| `ATLAS_LOGGING` | `ON` | Enable logging |
| `ATLAS_TESTS` | `ON` | Build tests (auto-disabled for CUDA) |
| `ATLAS_BENCHMARKS` | `ON` | Build benchmarks (auto-disabled for CUDA) |

> `ATLAS_USE_CUDA` and `ATLAS_USE_TBB` are **mutually exclusive** — exactly one must be `ON`.

See `CMakePresets.json` for the full list of available presets.

## Tests

Tests use GoogleTest and are only supported with TBB builds.

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --test-dir build/tbb-debug --output-on-failure
```

## Usage Example

```cpp
#include <atlas/atlas.h>
using namespace atlas;

int main() {
    using sim_t = float;

    // Domain
    const auto domain = system::Domain<sim_t>::builder()
        .with_geometry(geometry::Box<sim_t>::builder()
            .with_lower_corner(Vector3<sim_t>{-1.f, -1.f, -1.f})
            .with_upper_corner(Vector3<sim_t>{ 1.f,  1.f,  1.f})
            .make_host_shared())
        .with_cell_size(0.02f)
        .make_host_shared();

    // Spatial searcher
    const auto searcher = system::SpatialHashingSearcher<sim_t>::builder()
        .with_domain(domain)
        .with_range(system::NeighborSearchRange::single)
        .make_host_shared();

    // Codec
    const auto codec = system::SingleCodec<sim_t>::builder()
        .with_domain(domain)
        .make_host_shared();

    // Fluid (particle species)
    const auto fluid = system::Fluid<sim_t>::builder()
        .add_particle(FluidicParticle<sim_t>::builder()
            .with_molecular_mass(4.65e-26)
            .make_host_shared())
        .make_host_shared();

    return EXIT_SUCCESS;
}
```

## Project Structure

```
atlas-core-dev/
├── include/atlas/          # Header-only core library
│   ├── atlas.h             # Umbrella header (auto-generated)
│   ├── buffer/             # DeviceBuffer, HostBuffer
│   ├── codec/              # Particle encoding/decoding
│   ├── core/               # Portability macros
│   ├── domain/             # Simulation domain
│   ├── geometry/           # Geometric primitives (Box, Sphere, ...)
│   ├── math/               # Vector, Matrix, Quaternion
│   ├── memory/             # Smart pointers (host/device)
│   ├── parallel/           # parallel_for, parallel_sort, ...
│   ├── searcher/           # Spatial hashing neighbor search
│   ├── spatial/            # AABB, BVH, Ray tracing
│   └── ...
├── src/
│   ├── logging/            # Compiled logging backend
│   └── vizkit/             # OpenGL visualization (optional)
├── tests/                  # GoogleTest test suite
├── examples/
│   ├── solver/             # Particle simulation example
│   └── dynamic/            # Vizkit visualization example
├── external/               # In-tree third-party libs
│   ├── googletest/
│   ├── googlebenchmark/
│   ├── tinyobj/            # OBJ mesh loading
│   └── lyra/               # CLI argument parsing
└── tools/                  # Code generation & maintenance scripts
```

## License

See repository for license details.
