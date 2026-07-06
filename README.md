<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine Dev

Atlas is a C++20 particle simulation engine compiled entirely with **nvcc**, using a single `float` scalar type, with the TBB (CPU) or CUDA (GPU) backend selected through the Thrust device system, optional serialization support, and optional nanobind-based Python bindings.

The whole public API is aggregated through one umbrella header:

```cpp
#include <atlas/atlas.h>
```

All public types live under a single `atlas::` namespace (e.g. `atlas::Fluid`,
`atlas::Universe`, `atlas::Float3`, `atlas::Box`); enum values use `snake_case`
(`atlas::SpawnType::volume`, `atlas::MaterialType::molecule`).

## Features

- single-toolchain (nvcc) build with a `float` scalar simulation core
- TBB and CUDA Thrust device systems from one source tree — a GPU is needed only to **run** the CUDA build, never to build it
- active-prefix particle storage for source emission and sink compaction
- material records and Maxwell/thermal particle generators
- analytic geometry: box, sphere, cylinder, plane, circle, square, triangle, and triangle mesh
- spatial primitives and acceleration structures: rays, AABBs, spatial hashing, BVH, LBVH, and SAH BVH
- source, sink, collider, orchestrator, observer, and serialization runtime modules
- DSMC and SPH solver modules
- optional nanobind-based Python bindings, packaged as self-contained TBB and CUDA wheels (see [Python Bindings](#python-bindings))

## Requirements

The reference development environment is the Docker `dev` image (see
[Quick Start](#quick-start)); the requirements below apply to bare-metal setups.

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

Other in-tree dependencies (all vendored under [`external/`](external/)):
`tinyobjloader` (OBJ mesh loading), `lyra` (CLI parsing), `googletest`,
`googlebenchmark`, `protobuf` (binary snapshot serialization), and `nanobind`.

## Quick Start

### Docker (recommended)

Build the development image and work inside it — nvcc, TBB, CMake, and Ninja
are preinstalled:

```bash
docker build --target dev -t atlas-dev .
docker run --rm -it -v "$PWD":/workspace atlas-dev
```

### Build and test

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

### Bare-metal Linux prerequisites

```bash
sudo apt-get update
sudo apt-get install -y ninja-build libtbb-dev
# plus the CUDA 12.x toolkit (nvcc) from NVIDIA's repositories
```

## Build Presets

| Kind | TBB | CUDA |
|---|---|---|
| Configure | `tbb-debug`, `tbb-release` | `cuda-debug`, `cuda-release` |
| Build | `build-tbb-debug`, `build-tbb-release` | `build-cuda-debug`, `build-cuda-release` |
| Test | `ctest-tbb-debug` | `ctest-cuda-debug` |

The `*-release` presets build the engine and Python module only (tests are
excluded); run tests from the `*-debug` presets.

## Important CMake Options

| Option | Meaning |
|---|---|
| `ATLAS_DEVICE_SYSTEM` | Thrust device system: `TBB` (CPU, default) or `CUDA` (GPU) |
| `ATLAS_LOGGING` | Build logging support |
| `ATLAS_GOOGLE_TEST` | Build GoogleTest-based C++ tests |
| `ATLAS_PYTHON` | Build the nanobind Python bindings |
| `ATLAS_BENCHMARKS` | Build the runnable simulation / benchmark cases |

Constraints:

- TBB and the CUDA toolkit are required in every configuration
- benchmarks are currently TBB-only (disabled when `ATLAS_DEVICE_SYSTEM=CUDA`)

## Running a Simulation

The ready-to-run simulation programs are the cases under
[`benchmarks/atlas/`](benchmarks/atlas/). Each one is a complete end-to-end
setup — material and particle generators, a `Universe` that owns the boundary
units, a `Source` / `Sink` / `Collider`, and a DSMC or SPH solver — driven by
`System::update()`.

They are TBB-only and off by default. Enable them at configure time, build the
case you want, and run the produced executable:

```bash
cmake --preset tbb-debug -DATLAS_BENCHMARKS=ON
cmake --build build/tbb-debug --target atlas_benchmark_inflow -j$(nproc)
./build/tbb-debug/benchmarks/atlas/atlas_benchmark_inflow
```

Each case is a Google Benchmark executable: running it steps the simulation in a
timed loop and prints per-step timings. Pass `--help` for benchmark options
(iteration count, output format, filtering).

| Target | Source | Scenario |
|---|---|---|
| `atlas_benchmark_inflow` | [`inflow/`](benchmarks/atlas/inflow/) | source-driven DSMC particle inflow |
| `atlas_benchmark_waterfall` | [`waterfall/`](benchmarks/atlas/waterfall/) | waterfall-style particle scenario |
| `atlas_benchmark_cylinder_continum` | [`cylinder/continum/`](benchmarks/atlas/cylinder/continum/) | DSMC cylinder flow (continuum) |
| `atlas_benchmark_cylinder_rarefied_gas` | [`cylinder/rarefied_gas/`](benchmarks/atlas/cylinder/rarefied_gas/) | DSMC cylinder flow (rarefied gas) |

To see how a simulation is wired up in code — builders, boundary units, and the
solver pipeline — read a case's `main.cu`, e.g.
[`inflow/main.cu`](benchmarks/atlas/inflow/main.cu).

## Python Bindings

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

## Tests

Tests live under [`tests/`](tests/) and use GoogleTest directly. C++ tests are
discovered recursively from `tests/*.cpp` into an aggregate `atlas_tests` binary
plus per-directory `atlas_tests_<directory>` binaries. Run them from a `*-debug`
preset:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

## Documentation

Deeper design and contributor material lives in the repository, not this file:

- [`docs/architecture/`](docs/architecture/) — *what the program is*: runtime
  objects and ownership, the per-step simulation pipeline, the CUDA/TBB backend
  model, the module map, and structural conventions.
- [`docs/guidelines/`](docs/guidelines/) — *how to work on it*: workflow, code
  style, build/test, and dependencies.
- [`CLAUDE.md`](CLAUDE.md) is the entry map into both directories.
- Generated API reference (Doxygen) lives under [`docs/doxygen/`](docs/doxygen/).

## License

Licensed under the GNU General Public License v3.0 (GPLv3) or later. See [`LICENSE`](LICENSE) for the full text.
