<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine Dev

[![TBB CI](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb.yml/badge.svg?branch=main)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/tbb.yml?query=branch%3Amain)

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
the examples, and the benchmark cases. The release presets turn all five off and
build the engine alone.

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
| `ATLAS_EXAMPLES` | `ON` | `OFF` | Build the C++ example programs (`examples/cpp/`) |
| `ATLAS_BENCHMARKS` | `ON` | `OFF` | Build the Google Benchmark cases (`benchmarks/atlas/`) |

TBB is required in every configuration. The CUDA toolkit is required only when
nvcc compiles the sources.

## Running a Simulation

The ready-to-run simulation programs are the C++ examples under
[`examples/cpp/`](examples/cpp/). Each case is a `main.cu` — a standalone
simulation driven straight through the Atlas API, with no framework, printing its
own timings — built as `atlas_example_<case>` when `ATLAS_EXAMPLES` is on.

Examples are on in the debug presets. Build a case and run it:

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug --target atlas_example_cylinder -j$(nproc)
./build/tbb-debug/examples/cpp/atlas_example_cylinder
```

| Case | Source | Scenario |
|---|---|---|
| `cylinder` | [`cylinder/main.cu`](examples/cpp/cylinder/main.cu) | rarefied N₂ crossflow over a cylinder mesh at Kn ≈ 0.05 |

`atlas_example_cylinder [steps] [assets_dir] [output_dir]` loads
`assets/cylinder.obj`, prints the freestream regime it resolved to, steps the
`System`, and writes per-step CSV under `<output_dir>/data/`.

To see how a simulation is wired up in code — builders, boundary units, the
solver, and the observer — read
[`cylinder/main.cu`](examples/cpp/cylinder/main.cu).

Google Benchmark cases live separately under
[`benchmarks/atlas/`](benchmarks/atlas/) as `main.cpp` files built into
`atlas_benchmark_<case>_gbench` when `ATLAS_BENCHMARKS` is on. Only a `smoke`
case is wired for now — the representative benchmarks are to be rewritten on
Google Benchmark.

## Python Bindings

Build the module in-tree (the debug presets already enable it):

```bash
cmake --preset tbb-debug
cmake --build build/tbb-debug --target atlas_python -j$(nproc)
```

The module is a full assembly API that mirrors the C++ builders: every builder is
exposed as a lower-case factory function returning a ready object, and
`atlas.build_system(fluid, universe, dt, solver=, source=, generator=, colliders=,
sinks=, codec=, observer=)` wires them into a runnable `System`. Initial particle
state is seeded from numpy with `fluid_from_arrays(positions, velocities, ...)`, and
`load_fluid` / `load_universe` reload snapshots. The `System` exposes
`update()` and `save()`, the scalars `step`, `dt`, `particle_count`, and
`cell_count`, and `positions()` / `velocities()` / `species()` returning numpy arrays.

```python
import atlas
import numpy as np

Vec = atlas.Float3

materials = atlas.material_dictionary([
    atlas.molecule(mass=4.65e-26, translational_energy=0.0, rotational_energy=0.0,
                   vibrational_energy=0.0, reference_diameter=4.17e-10,
                   reference_temperature=273.0, viscosity_index=0.74,
                   scattering_parameter=1.0),
])

rng = np.random.default_rng(0)
positions = rng.uniform(0.1, 0.9, size=(200, 3)).astype(np.float32)
velocities = (rng.standard_normal((200, 3)) * 300.0).astype(np.float32)

fluid = atlas.fluid_from_arrays(positions, velocities, statistical_weight=1e18, materials=materials)
universe = atlas.universe(Vec(0, 0, 0), Vec(1, 1, 1), cell_size=1.0)
solver = atlas.dsmc_solver(kernel_type=atlas.DsmcKernelType.variable_hard_sphere)

system = atlas.build_system(fluid=fluid, universe=universe, dt=1e-4, solver=solver)
for _ in range(20):
    system.update()

print(system.particle_count, np.linalg.norm(system.velocities(), axis=1).mean())
```

A runnable version is [`examples/python/dsmc_dense_cell.py`](examples/python/dsmc_dense_cell.py);
the full API reference is [`docs/guidelines/python.md`](docs/guidelines/python.md).

Self-contained, redistributable wheels (bundling libtbb / libcudart) are built
inside the packaging image. Initialise the submodules the build needs first:

```bash
git submodule update --init --recursive \
    external/nanobind external/tinyobj external/protobuf

docker build --target wheel -t atlas-wheel .

# ATLAS_WHEEL_BACKENDS=TBB builds a CPU-only wheel (no GPU or driver at run time);
# omit it to build both TBB and CUDA wheels.
docker run --rm -v "$PWD":/workspace -w /workspace \
    -e ATLAS_WHEEL_BACKENDS=TBB \
    atlas-wheel -c 'bash scripts/build_wheels.sh'
# -> dist/tbb/*.whl (and dist/cuda/*.whl when both backends are built)

pip install dist/tbb/*.whl
```

The full wheel flow — auditwheel repair, the `abi3` stable-ABI details, and building
without Docker — is in [`docs/guidelines/python.md`](docs/guidelines/python.md).

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

### GitHub Actions: TBB on `main`

The [TBB CI workflow](.github/workflows/tbb.yml) runs automatically on pushes to
`main`. To run it manually, open **Actions → TBB CI → Run workflow** and select
`main`. Other branches and pull requests do not run this workflow; a manual run
on another branch skips its job.

The Ubuntu 22.04 runner uses the native C++ compiler and TBB, with no CUDA toolkit
or GPU. It builds and runs the aggregate GoogleTest suite, creates a Python source
distribution, builds a wheel from that archive, checks package metadata, installs
the wheel, and runs the Python DSMC example. Successful runs attach the
`tbb-linux-cp311` artifact, containing the source archive and
Python 3.11 wheel. This CI wheel requires the runner-compatible Linux/TBB runtime;
use the manylinux release wheels below for redistribution.

## Python Distribution

The distribution name is **`atlas-engine`** and the import name is **`atlas`**.
Packaging metadata and bundled source dependencies are configured in
[`pyproject.toml`](pyproject.toml). Source archives include the required git
submodule contents, so installing an archive does not require a git checkout.
Building from source still requires a C++20 compiler, CMake, Ninja, TBB, and
network access for build tools and Abseil.

The [Publish Python workflow](.github/workflows/publish-python.yml) runs manually on
`main` only. It builds a source archive and uses that archive to build and test
TBB manylinux x86_64 wheels for CPython 3.9–3.13. CUDA wheels remain a separate
local build through `scripts/build_wheels.sh`; they are not uploaded to PyPI.

To prepare or publish a version:

1. Update `project.version` in `pyproject.toml` and the matching version in
   `CITATION.cff`, then push the changes to `main`. Packaged
   `atlas.__version__` follows the Python distribution version automatically.
2. Open **Actions → Publish Python → Run workflow**, selecting `main`.
   Leave **Publish the validated distributions to PyPI** unchecked to download
   the `python-sdist` and `python-wheels` artifacts without publishing.
3. For actual publication, configure a PyPI Trusted Publisher for this repository,
   workflow **`publish-python.yml`**, and GitHub environment **`pypi`**. Create the
   matching GitHub environment, then run the workflow on `main` with the publish
   checkbox enabled. Each uploaded version must be new on PyPI.

Tag pushes and GitHub releases do not trigger Python publication. The Zenodo
release/DOI workflow described below remains separate. Once a version has been
published, install its CPU package with:

```bash
python -m pip install atlas-engine
python -c "import atlas; print(atlas.__version__)"
```

See the official [GitHub branch-filter documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushbranchestagsbranches-ignoretags-ignore)
and [PyPI Trusted Publishing guide](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Documentation

Deeper design and contributor material lives in the repository, not this file:

- [`docs/atlas/`](docs/atlas/) — *per-module documentation*: the tagged-union
  leaf pattern and how to extend each module. There is a document for every
  module under `include/atlas/`.
- [`docs/guidelines/`](docs/guidelines/) — *how to work on it*: workflow, code
  style, build/test, dependencies, and the
  [Python bindings guide](docs/guidelines/python.md). Start at
  [`docs/guidelines/README.md`](docs/guidelines/README.md).
- [`AGENTS.md`](AGENTS.md) is the entry map into both directories and the
  repository-wide instruction file for AI coding agents.
- Every header and source under `include/atlas/` and `src/atlas/` carries Doxygen
  comments; the convention is recorded in
  [`coding-style.md` §7](docs/guidelines/coding-style.md).

## Citation and Zenodo DOI

Citation metadata is maintained in [`CITATION.cff`](CITATION.cff), which is
supported by both GitHub and Zenodo. Its author, version, and license match
the project metadata in [`pyproject.toml`](pyproject.toml).

To publish a release with a public DOI:

1. Sign in to [Zenodo](https://zenodo.org/) with GitHub and enable
   `Wontae-Lee/atlas-engine-dev` in the GitHub integration settings.
2. Review the authors in `CITATION.cff`, adding any coauthors and known ORCID
   identifiers or affiliations. Update `version` to match the release and add
   `date-released` in `YYYY-MM-DD` format when the release date is known.
3. Commit and push the metadata, then publish a GitHub release whose tag
   includes that commit. For the current version, the tag would be `v0.1.0`.
4. Wait for Zenodo to process the release and check the published record's
   metadata and DOI. Adding these files alone does not register a DOI.
5. Add the issued release DOI to `CITATION.cff` as `doi` and link the Zenodo
   record here. Keep the DOI, version, and release date referring to the same
   release; do not carry a previous release's DOI into a new release.

See Zenodo's [citation metadata guide](https://help.zenodo.org/docs/github/describe-software/citation-file/)
and [GitHub release archiving guide](https://help.zenodo.org/docs/github/archive-software/github-upload/).
`CITATION.cff` is sufficient for this workflow; if a `.zenodo.json` file is
added later, Zenodo will use that file instead of `CITATION.cff`.

## License

Licensed under the GNU General Public License v3.0 (GPLv3) or later. See [`LICENSE`](LICENSE) for the full text.
