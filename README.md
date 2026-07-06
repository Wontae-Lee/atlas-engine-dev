<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine Dev

Atlas is a C++20 particle simulation engine compiled entirely with **nvcc**, using a single `float` scalar type, with the TBB (CPU) or CUDA (GPU) backend selected through the Thrust device system, optional serialization support, and optional nanobind-based Python bindings.

The public API is aggregated through:

```cpp
#include <atlas/atlas.h>
```

## Overview

Atlas is built around a small set of runtime objects (all under `atlas::`):

- `Fluid` owns particle storage, material properties, particle generators, observers, and registered fluid states
- `Universe` owns the Cartesian simulation domain, grid resolution, the per-role boundary unit fields, observers, and registered universe states
- `Source` emits particles into inactive capacity behind the fluid active prefix
- `Sink` removes particles and compacts survivors back into a dense active prefix
- `Collider` reflects particles off boundary units with a surface interaction model
- `Orchestrator` coordinates search, codec, measurement, field forces, gravity, and solver stages
- `System` is the top-level step driver

`System::update()` runs the current simulation step in this order:

1. `source->update(dt)`
2. `orchestrator->update(dt)`
3. `collider->update(dt)` when a collider is installed, otherwise fallback time integration
4. `sink->update(dt)`

When no collider is installed, fallback motion is:

```cpp
position += velocity * dt;
```

## Backend Model

Every translation unit is compiled as CUDA by nvcc. The backend is the Thrust
device system, selected with one option at configure time:

- `ATLAS_DEVICE_SYSTEM=TBB` — CPU build; Thrust device system = TBB, no GPU code generated
- `ATLAS_DEVICE_SYSTEM=CUDA` — GPU build; Thrust device system = CUDA

The Thrust host system is always TBB. A GPU is required only to **run** the
CUDA variant, never to build it.

Atlas keeps most runtime APIs backend-agnostic through shared aliases:

- `DeviceBuffer<T>` holds data the active device system processes (`thrust::device_vector<T>` on CUDA)
- `HostBuffer<T>` is host-resident staging/config data
- `device_shared_ptr<T>` maps to `std::shared_ptr<T>` on TBB and CUDA-aware managed ownership on CUDA
- `parallel_for<ExecutionPolicy>(...)` dispatches to the active backend implementation

Backend-specific compile-time paths use `ATLAS_TASKING_TBB` or `ATLAS_TASKING_CUDA`.

## Key Concepts

### Active Prefix Storage

Particle storage separates allocated capacity from the logical particle population:

- `Fluid::buffer_size()` is total allocated capacity
- `Fluid::particle_count()` is the active dense prefix length
- active particles live in `[0, particle_count)`
- inactive capacity begins at `particle_count`

This contract matters for source and sink behavior:

- `Source` appends into inactive capacity without assuming capacity equals population
- `Sink` compacts surviving particles back into `[0, particle_count)`
- tests should set `particle_count` explicitly when active particles should exist

### Universe-Owned Boundary Units

Boundary units (the geometry a source spawns from, a sink despawns at, or a
collider reflects off) are owned centrally by the `Universe`, one field per
consumer role:

- register units with `Universe::Builder::with_{source,sink,collider,measurer}_units(...)`
- `Source` / `Sink` / `Collider` / `VolumeMeasurer` take the domain via
  `with_universe(...)` and borrow the matching unit field — they do not own units

### Builder Pattern

Most public types use nested builders with fluent `.with_*()` setters and `.build()` / `.make_host_shared()` endpoints.

```cpp
const auto fluid = atlas::Fluid::builder()
    .with_buffer_size(4096)
    .with_properties(properties)
    .with_generators(generators)
    .make_host_shared();

const auto system = atlas::System::builder()
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

- particle capacity belongs to `Fluid` (`with_buffer_size(...)`), not `System`
- boundary units belong to `Universe`; `Source`/`Sink`/`Collider` borrow via `with_universe(...)`

### Namespaces

All public types, aliases, and enums are re-exported under a single `atlas::`
namespace (e.g. `atlas::Fluid`, `atlas::Universe`, `atlas::Vector3`,
`atlas::Box`, `atlas::SpawnType`). Enum values use `snake_case`
(`atlas::SpawnType::volume`, `atlas::MaterialType::molecule`).

## Repository Layout

```text
atlas-engine-dev/
├── include/atlas/            # Public headers (declarations + inline device code)
│   ├── atlas.h               # Generated umbrella header
│   ├── buffer/     codec/     collider/  container/  core/     fluid/
│   ├── generator/  geometry/  indexer/   iterator/   logging/  material/
│   ├── math/       measure/   memory/    observer/   orchestrator/
│   ├── parallel/   random/    remove/    sampling/   scan/     searcher/
│   ├── serialization/  shuffle/  sink/    solver/     source/   spatial/
│   ├── sync/       system/    transform/ tuple/      unit/     universe/
├── src/atlas/                # Engine definitions (.cu), mirrors include/atlas/
├── src/logging/              # Compiled logging implementation
├── src/serialization/        # Snapshot serialization support
├── src/python/atlas/         # Optional nanobind Python bindings
├── assets/                   # Mesh assets (OBJ/MTL)
├── docs/                     # Logo, Doxygen, architecture + guideline docs
├── tests/                    # GoogleTest sources
├── benchmarks/               # Benchmark cases and reference code
├── external/
└── tools/
```

## Features

- single-toolchain (nvcc) build with `float` scalar simulation core
- TBB and CUDA Thrust device systems from one source tree
- active-prefix particle storage for source emission and sink compaction
- material records and Maxwell/thermal particle generators
- analytic geometry including box, sphere, cylinder, plane, circle, square, triangle, and triangle mesh
- spatial primitives and acceleration structures including rays, AABBs, spatial hashing, BVH, LBVH, and SAH BVH
- source, sink, collider, orchestrator, observer, and serialization runtime modules
- DSMC and SPH solver modules
- optional nanobind-based Python bindings, packaged as self-contained TBB and CUDA wheels (see [Python Bindings](#python-bindings))

## Requirements

The reference development environment is the Docker `dev` image (see
[Quick Start](#quick-start)); the requirements below apply to bare-metal
setups.

### Core (always required)

| Dependency | Notes |
|---|---|
| CUDA Toolkit 12.x | nvcc compiles every translation unit; also provides Thrust |
| TBB | Thrust host system in every configuration |
| CMake 3.20+ | Presets are provided in [`CMakePresets.json`](CMakePresets.json) |
| C++20 host compiler | GCC 11+ is a reasonable target |
| Ninja | All presets use Ninja |

### GPU execution only

| Dependency | Notes |
|---|---|
| NVIDIA driver | Needed only to run `ATLAS_DEVICE_SYSTEM=CUDA` builds |

### Python Bindings

| Dependency | Notes |
|---|---|
| Python 3.8+ | Interpreter with development headers (`Development.Module`) |

Enabled with `ATLAS_PYTHON=ON`. nanobind itself is vendored under `external/`.

### In-tree Dependencies

All vendored under [`external/`](external/):

- `tinyobjloader` — OBJ mesh loading
- `lyra` — CLI argument parsing
- `googletest` — C++ unit test framework
- `googlebenchmark` — microbenchmark framework
- `protobuf` — binary snapshot serialization
- `nanobind` — Python bindings

## Build Presets

| Kind | TBB | CUDA |
|---|---|---|
| Configure | `tbb-debug`, `tbb-release` | `cuda-debug`, `cuda-release` |
| Build | `build-tbb-debug`, `build-tbb-release` | `build-cuda-debug`, `build-cuda-release` |
| Test | `ctest-tbb-debug` | `ctest-cuda-debug` |

The `*-release` presets build the engine and Python module only (tests are
excluded); run tests from the `*-debug` presets.

## Quick Start

### Docker (recommended)

Build the development image and work inside it — nvcc, TBB, CMake, and Ninja
are preinstalled:

```bash
docker build --target dev -t atlas-dev .
docker run --rm -it -v "$PWD":/workspace atlas-dev
```

Inside the container (or on a bare-metal host with the CUDA toolkit and TBB
installed):

CPU (TBB device system) debug build:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

GPU (CUDA device system) debug build — a GPU is needed only to run the result:

```bash
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)
```

### Bare-metal Linux

```bash
sudo apt-get update
sudo apt-get install -y ninja-build libtbb-dev
# plus the CUDA 12.x toolkit (nvcc) from NVIDIA's repositories
```

### Python Bindings

Build the module in-tree:

```bash
cmake --preset tbb-debug -DATLAS_PYTHON=ON
cmake --build build/tbb-debug --target atlas_python -j$(nproc)
```

Or build self-contained, redistributable wheels (bundling libtbb / libcudart)
inside the packaging image:

```bash
docker build --target wheel -t atlas-wheel .
docker run --rm -v "$PWD":/workspace -w /workspace atlas-wheel \
    -c 'bash scripts/build_wheels.sh'
# -> dist/tbb/*.whl and dist/cuda/*.whl
```

## Important CMake Options

| Option | Meaning |
|---|---|
| `ATLAS_DEVICE_SYSTEM` | Thrust device system: `TBB` (CPU, default) or `CUDA` (GPU) |
| `ATLAS_LOGGING` | Build logging support |
| `ATLAS_GOOGLE_TEST` | Build GoogleTest-based C++ tests |
| `ATLAS_PYTHON` | Build the nanobind Python bindings |
| `ATLAS_BENCHMARKS` | Build benchmarks |

Constraints:

- TBB and the CUDA toolkit are required in every configuration
- benchmarks are currently TBB-only (disabled when `ATLAS_DEVICE_SYSTEM=CUDA`)

## Minimal Runtime Example

The current runtime model is `Fluid + Universe (owning boundary units) +
optional subsystems + System`.

```cpp
#include <atlas/atlas.h>

int main() {
    const auto observer = atlas::Observer::builder()
        .with_source_sensor_metrics(1024)
        .with_sink_sensor_metrics(1024)
        .make_host_shared();

    // Material + generator for the fluid.
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    properties[0] = atlas::MaterialProperties::builder()
        .with_type(atlas::MaterialType::molecule)
        .with_molecular_mass(4.651734e-26f)
        .with_reference_diameter(4.17e-10f)
        .build();

    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);
    generators[0] = atlas::MaxwellBoltzmannGenerator::builder()
        .with_temperature(300.0f)
        .with_molecular_mass(4.651734e-26f)
        .with_bulk_velocity(atlas::Vector3(0, 0, 0))
        .with_seed(42u)
        .make_host_shared();

    const auto fluid = atlas::Fluid::builder()
        .with_buffer_size(4096)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();

    // A boundary unit = geometry + a placement (Sync). Units are owned by the
    // Universe, one field per role (here: source units).
    const auto source_geometry = atlas::Geometry(*atlas::Box::builder()
        .with_lower_corner(atlas::Vector3(-0.8f, -0.2f, -0.2f))
        .with_upper_corner(atlas::Vector3(-0.4f, 0.2f, 0.2f))
        .make_host_shared());

    const auto sync = atlas::Sync::builder()
        .with_rigid_pose(atlas::Vector3(0, 0, 0), atlas::Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();

    atlas::HostBuffer<atlas::Unit> source_units(1);
    source_units[0] = atlas::Unit::builder()
        .with_geometry(source_geometry)
        .with_sync(sync)
        .build();

    const auto universe = atlas::Universe::builder()
        .with_lower_corner(atlas::Vector3(-1, -1, -1))
        .with_upper_corner(atlas::Vector3(1, 1, 1))
        .with_cell_size(0.1f)
        .with_source_units(source_units)
        .make_host_shared();

    // The Source borrows the Universe's source-unit field.
    const auto source = atlas::Source::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_observer(observer)
        .with_spawn_types(atlas::HostBuffer<atlas::SpawnType>(1, atlas::SpawnType::volume))
        .with_spawn_operator(atlas::Spawn(atlas::SpawnType::volume))
        .with_spacing(0.05f)
        .make_host_shared();

    const auto system = atlas::System::builder()
        .with_fluid(fluid)
        .with_domain(universe)
        .with_source(source)
        .with_dt(0.01f)
        .make_host_shared();

    system->update();
    return 0;
}
```

For a complete end-to-end setup with DSMC, collider interaction, sink removal, and observer export, see the benchmark cases under [`benchmarks/atlas/`](benchmarks/atlas/).

## Tests

Tests live under [`tests/`](tests/) and use GoogleTest directly:

```cpp
#include <gtest/gtest.h>
```

Current test structure:

- C++ tests are discovered recursively from `tests/*.cpp`
- aggregate C++ binary: `atlas_tests`
- per-directory C++ binaries: `atlas_tests_<directory>`

Recommended TBB test run:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

## Benchmark Cases

The reference simulation programs live under
[`benchmarks/atlas/`](benchmarks/atlas/) and are built when
`ATLAS_BENCHMARKS=ON`:

- [`benchmarks/atlas/cylinder/`](benchmarks/atlas/cylinder/) demonstrates DSMC cylinder flow, collider interaction, sink removal, and observer export
- [`benchmarks/atlas/honeycomb/`](benchmarks/atlas/honeycomb/) demonstrates honeycomb channel flow with mesh-based geometry
- [`benchmarks/atlas/inflow/`](benchmarks/atlas/inflow/) demonstrates source-driven particle inflow
- [`benchmarks/atlas/intake/`](benchmarks/atlas/intake/) demonstrates intake-style source and sink setup
- [`benchmarks/atlas/waterfall/`](benchmarks/atlas/waterfall/) demonstrates a waterfall-style particle scenario

Each case has a single `main.cu` entry point; nvcc compiles it for both device systems.

## Development Notes

- use portability macros from [`include/atlas/core/macros.h`](include/atlas/core/macros.h): `ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE`, `ATLAS_FORCE_INLINE`, `ATLAS_NODISCARD`, `ATLAS_MAYBE_UNUSED`, and `RESTRICT`
- headers declare the API and hold inline device-callable code; host-only definitions live in `src/atlas/**/*.cu`, mirroring the header layout
- do not hand-edit generated umbrella headers such as [`include/atlas/atlas.h`](include/atlas/atlas.h); use [`tools/generate_headers.py`](tools/generate_headers.py)
- a public rename is a breaking change; rename public spellings only when explicitly requested
- the Python bindings under `src/python/atlas/` are built only when `ATLAS_PYTHON=ON`
- contributor docs live under [`docs/architecture/`](docs/architecture/) (what the program is) and [`docs/guidelines/`](docs/guidelines/) (how to work on it); see [`CLAUDE.md`](CLAUDE.md)

## License

See the repository license files and project metadata for licensing terms.
