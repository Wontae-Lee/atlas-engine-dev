<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine Dev

Atlas is a C++20 particle simulation engine with a single `float` scalar type, a TBB (CPU) or CUDA (GPU) backend chosen at configure time, protobuf snapshot serialization, and optional nanobind-based Python bindings. The CPU build needs **neither nvcc nor the CUDA toolkit**.

The engine targets **large domains, simply and fast**. Where physical fidelity
trades against speed or simplicity, Atlas takes the simple option — and says so
where it does.

The whole public API is aggregated through one umbrella header:

```cpp
#include <atlas/atlas.h>
```

All public types live under a single `atlas::` namespace (e.g. `atlas::Fluid`,
`atlas::Universe`, `atlas::Float3`, `atlas::Box`); enum values use `snake_case`
(`atlas::SourceType::volume`, `atlas::MaterialType::molecule`).

## Features

- one source tree, two backends: a plain host-compiler build on `std::vector` + TBB, or an nvcc build on Thrust + CUDA
- a GPU is needed only to **run** the CUDA build, never to build it
- a six-stage step pipeline on one object: `System::update()` runs emit → search → allocate → solve → advect → remove
- particle compaction in `Fluid`: sinks mark survivors, one scan rebuilds every particle array
- material records (`Molecule`, `Atom`, `Ion`, `Neutron`, `Solid`) and four particle generators
- analytic geometry: box, sphere, cylinder, plane, circle, square, triangle, and triangle mesh
- spatial primitives and acceleration structures: rays, AABBs, spatial hashing, BVH, LBVH, and SAH BVH
- DSMC solver with hard-sphere, VHS, and VSS collision kernels
- source, sink, collider, codec, searcher, observer, and serialization runtime modules
- CSV observation and protobuf snapshot save / restart
- optional nanobind-based Python bindings, packaged as self-contained TBB and CUDA wheels (see [Python Bindings](#python-bindings))

### The tagged-union leaf pattern

Most runtime modules share one shape: a concrete umbrella type wraps one of
several self-contained leaf types and dispatches to it. No virtual dispatch,
because these types are captured by value into device lambdas.

Which umbrella a module uses is decided by one question — **does the leaf own a
`DeviceBuffer`?**

| | Modules | Umbrella |
|---|---|---|
| Trivially copyable leaves | `Geometry`, `Material`, `Collider`, `Sink`, `DsmcKernel` | [`DeviceVariant`](include/atlas/core/device_variant.h) — host+device, copy-based |
| Leaves owning a `DeviceBuffer` | `Source`, `Generator`, `Codec` | [`HostVariant`](include/atlas/core/host_variant.h) — host-only, move-based |

`Solver` is the exception: an abstract base class, because a solver's `solve()`
runs on the host and launches its own kernels.

An object that owns `DeviceBuffer`s cannot itself enter a kernel, so each hands
out a trivially-copyable **view** of raw pointers instead:
`SpatialHashingSearcherView`, `FluidDsmcView`, `UniverseDsmcView`,
`TriangleMeshView`, `BvhView`.

### The backend split

CMake defines exactly one of `ATLAS_BACKEND_CUDA` and `ATLAS_BACKEND_TBB`. Only
ten headers — under [`buffer/`](include/atlas/buffer/),
[`memory/`](include/atlas/memory/), [`parallel/`](include/atlas/parallel/), and
[`scan/`](include/atlas/scan/) — branch on it. The other ~120 files name
`DeviceBuffer` and `parallel_for` and never learn the difference.

| | CUDA backend | Host backend |
|---|---|---|
| `HostBuffer<T>` / `DeviceBuffer<T>` | `thrust::host_vector` / `thrust::device_vector` | `std::vector` / `std::vector` |
| `parallel_for`, `parallel_fill`, `parallel_sort` | Thrust | TBB |
| `exclusive_scan` | `thrust::exclusive_scan` | `std::exclusive_scan` |
| `default_random_engine`, `uniform_real_distribution` | one shared implementation | one shared implementation |

The RNG is written out rather than aliased so both backends draw from the
identical stream — a case reproduced on the CPU has to match the GPU run it
checks. The engine holds no `__global__`, no `<<<>>>`, and no
`cudaMalloc` outside `memory.h`; every kernel is a `parallel_for` over an
`ATLAS_ALL_DEVICE` lambda, and those annotations vanish outside `__CUDACC__`.
That is why the `.cu` sources compile as ordinary C++.

## Requirements

The reference development environment is the Docker `dev` image (see
[Quick Start](#quick-start)); the requirements below apply to bare-metal setups.

### Core (always required)

| Dependency | Notes |
|---|---|
| TBB | Backs the host-side parallel algorithms in every configuration |
| CMake 3.20+ | Presets are provided in [`CMakePresets.json`](CMakePresets.json) |
| C++20 host compiler | GCC 11+ is a reasonable target |
| Ninja | All presets use Ninja |

### GPU builds only

| Dependency | Notes |
|---|---|
| CUDA Toolkit 12.x | nvcc compiles every translation unit and provides Thrust. Needed for `ATLAS_DEVICE_SYSTEM=CUDA`, or for `ATLAS_HOST_COMPILER=nvcc` |
| NVIDIA driver | Needed only to **run** `ATLAS_DEVICE_SYSTEM=CUDA` builds |

### Python Bindings

| Dependency | Notes |
|---|---|
| Python 3.8+ | Interpreter with development headers (`Development.Module`) |

nanobind is vendored under `external/`, but it carries its own submodule, so a
clone that skipped `--recursive` must run this before `ATLAS_PYTHON=ON` will
configure:

```bash
git submodule update --init --recursive external/nanobind
```

Other in-tree dependencies (all vendored under [`external/`](external/)):
`tinyobjloader` (OBJ mesh loading), `googletest`, `googlebenchmark`, and
`protobuf` (binary snapshot serialization). protobuf pulls Abseil in at
configure time.

## Quick Start

### Docker (recommended)

Build the development image and work inside it — the CUDA toolkit, TBB, CMake,
and Ninja are preinstalled:

```bash
docker build --target dev -t atlas-dev .
docker run --rm -it -v "$PWD":/workspace atlas-dev
```

### Build and test

Inside the container, or on a bare-metal host. The CPU build needs only TBB and
a C++20 compiler:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

GPU build — needs the CUDA toolkit; a GPU only to run the result:

```bash
cmake --preset cuda-debug
cmake --build build/cuda-debug -j$(nproc)
```

### Bare-metal Linux prerequisites

```bash
sudo apt-get update
sudo apt-get install -y ninja-build libtbb-dev
# that is everything the tbb-* presets need.
# the cuda-* presets additionally need the CUDA 12.x toolkit from NVIDIA's repositories.
```

## Build Presets

| Kind | TBB (host compiler) | CUDA (nvcc) |
|---|---|---|
| Configure | `tbb-debug`, `tbb-release` | `cuda-debug`, `cuda-release` |
| Build | `build-tbb-debug`, `build-tbb-release` | `build-cuda-debug`, `build-cuda-release` |
| Test | `ctest-tbb-debug` | `ctest-cuda-debug` |

The debug presets turn **everything** on — logging, tests, the Python module,
and the benchmark cases. The release presets turn all four off and build the
engine alone.

`tbb-nvcc-debug` builds the CPU backend *through nvcc*. It exists for CI: nvcc
rejects constructs the host compiler accepts — notably an extended
`__host__ __device__` lambda inside a private member function — and a
host-compiler-only build would stop catching them.

## Important CMake Options

| Option | Debug presets | Release presets | Meaning |
|---|---|---|---|
| `ATLAS_DEVICE_SYSTEM` | — | — | Parallel backend: `TBB` (CPU, std + TBB) or `CUDA` (GPU, nvcc + Thrust) |
| `ATLAS_HOST_COMPILER` | `native` | `native` | Compiler for a TBB build: `native` or `nvcc` |
| `ATLAS_LOGGING` | `ON` | `OFF` | Build logging support (`atlas::warn`, …) |
| `ATLAS_GOOGLE_TEST` | `ON` | `OFF` | Build GoogleTest-based C++ tests |
| `ATLAS_PYTHON` | `ON` | `OFF` | Build the nanobind Python bindings |
| `ATLAS_BENCHMARKS` | `ON` | `OFF` | Build the runnable simulation / benchmark cases |

TBB is required in every configuration. The CUDA toolkit is required only when
nvcc compiles the sources.

## Running a Simulation

The ready-to-run simulation programs are the cases under
[`benchmarks/atlas/`](benchmarks/atlas/). Each case directory holds up to two
entry points:

| File | Target | What it is |
|---|---|---|
| `main.cu` | `atlas_benchmark_<case>` | a standalone simulation driven straight through the Atlas API — no benchmark framework, prints its own timings |
| `main.cpp` | `atlas_benchmark_<case>_gbench` | the same case wrapped in Google Benchmark |

Both are optional; CMake creates a target only for the file that exists.

Benchmarks are on in the debug presets. Build a case and run it:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug --target atlas_benchmark_cylinder -j$(nproc)
./build/tbb-debug/benchmarks/atlas/atlas_benchmark_cylinder
```

| Case | Source | Scenario |
|---|---|---|
| `cylinder` | [`cylinder/main.cu`](benchmarks/atlas/cylinder/main.cu) | rarefied N₂ crossflow over a cylinder mesh at Kn ≈ 0.05 |

`atlas_benchmark_cylinder [steps] [assets_dir] [output_dir]` loads
`assets/cylinder.obj`, prints the freestream regime it resolved to, steps the
`System`, and writes per-step CSV under `<output_dir>/data/`.

To see how a simulation is wired up in code — builders, boundary units, the
solver, and the observer — read
[`cylinder/main.cu`](benchmarks/atlas/cylinder/main.cu).

## Python Bindings

Build the module in-tree (the debug presets already enable it):

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug --target atlas_python -j$(nproc)
```

The bindings are currently a stub: `atlas.System` with `update`, `save`, `step`,
and `dt`, and nothing else. The old module bound types the engine restructuring
removed; it is being grown back one type at a time.

Self-contained, redistributable wheels (bundling libtbb / libcudart) are built
inside the packaging image:

```bash
docker build --target wheel -t atlas-wheel .
docker run --rm -v "$PWD":/workspace -w /workspace atlas-wheel \
    -c 'bash scripts/build_wheels.sh'
# -> dist/tbb/*.whl and dist/cuda/*.whl
```

## Tests

Tests live under [`tests/atlas/`](tests/atlas/), mirroring `include/atlas/` and
`src/atlas/`, and use GoogleTest directly. Sources are globbed recursively into
an aggregate `atlas_tests` binary plus one `atlas_tests_<module>` binary per
directory, so a new test file needs no CMake edit. Run them from a `*-debug`
preset:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug -j$(nproc)
ctest --preset ctest-tbb-debug
```

## Documentation

Deeper design and contributor material lives in the repository, not this file:

- [`docs/atlas/`](docs/atlas/) — *per-module documentation*: the tagged-union
  leaf pattern and how to extend each module. Covers codec, collider, material,
  sink, and source.
- [`docs/guidelines/`](docs/guidelines/) — *how to work on it*: workflow, code
  style, build/test, and dependencies. Start at
  [`docs/guidelines/README.md`](docs/guidelines/README.md).
- [`CLAUDE.md`](CLAUDE.md) is the entry map into both directories.
- Every header and source under `include/atlas/` and `src/atlas/` carries Doxygen
  comments; the convention is recorded in
  [`coding-style.md` §7](docs/guidelines/coding-style.md).

## License

Licensed under the GNU General Public License v3.0 (GPLv3) or later. See [`LICENSE`](LICENSE) for the full text.
