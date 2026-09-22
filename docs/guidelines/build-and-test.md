# Build and Test

The build/test policy, the CMake presets and options, and the rules for
authoring tests.

---

## 1. Run Policy

Do not run builds, tests, benchmarks, simulations, generators, or formatters
unless the user explicitly asks.

The Python bindings are packaged with scikit-build-core; when the user asks to
build them, use `scripts/build_wheels.sh` (it installs its own toolchain via
`pip`). The repo provisions no `.venv`.

---

## 2. Toolchain and CMake Presets

Two switches pick the toolchain.

`ATLAS_DEVICE_SYSTEM` selects the parallel backend. `CUDA` compiles the engine
and Python extension with nvcc and uses Thrust's containers and algorithms.
Host-only serialization, logging, and external dependencies use C/C++ compilers. `TBB`
(the default) uses neither: buffers are `std::vector`, the algorithms are TBB's,
and the host compiler builds the whole tree. The `.cu` suffix survives on the
engine sources because they hold no CUDA-only syntax — every kernel is a
`parallel_for` over an `ATLAS_ALL_DEVICE` lambda, and those annotations vanish
outside `__CUDACC__`.

`ATLAS_HOST_COMPILER=native` uses the selected host compiler for TBB;
`ATLAS_HOST_COMPILER=nvcc` forces a TBB build through nvcc.
The `tbb-nvcc-debug` preset catches constructs **nvcc rejects but the host compiler
accepts** — notably an extended `__host__ __device__` lambda inside a private
member function, which is the only reason `System::mark_survivors` and friends
are public. Native TBB CI does not catch those; the current workflows do not
run this optional nvcc preset.

A CUDA toolkit is therefore needed only when nvcc is in play. A GPU is needed
only to *run* the CUDA variant.

Docker provides separate development toolchains. Use `tbb-dev` for a native
GCC/G++ build without CUDA, or `cuda-dev` when nvcc is needed:

```bash
docker build --target tbb-dev -t atlas:tbb-dev .
docker run --rm -it -v "$PWD":/workspace atlas:tbb-dev
docker build --target cuda-dev -t atlas:cuda-dev .
docker run --rm -it --gpus all -v "$PWD":/workspace atlas:cuda-dev
```

The CUDA build itself does not need `--gpus all`; use it when running GPU code.
The `dev` target remains an alias of `cuda-dev`. The `tbb` and `cuda` runtime
targets have Python and Atlas installed and are intended to run user scripts.
See [docker.md](docker.md) for Ubuntu versions, runtime usage, and build options.

The project requires CMake 3.20+; the version-3 `CMakePresets.json` format
requires CMake 3.21+. The presets use Ninja. Important presets:

- Configure: `tbb-debug`, `tbb-release` (CPU; host compiler, no CUDA toolkit),
  `tbb-gcc-debug`, `tbb-gcc-release` (explicit `gcc`/`g++`),
  `tbb-nvcc-debug` (CPU backend built by nvcc),
  `cuda-debug`, `cuda-release` (GPU; nvcc + Thrust),
  `tbb-application-release`, `cuda-application-release` (native applications)
- Test: `ctest-tbb-debug`, `ctest-tbb-gcc-debug`, `ctest-tbb-nvcc-debug`, `ctest-cuda-debug`

The debug presets enable logging and GoogleTest while keeping Python, examples,
benchmarks, and interactive rendering off. Release presets produce the core
only. The two `*-application-release` presets add the native execution/rendering
targets and examples to their backend's release configuration. Python packaging
uses `pyproject.toml` rather than a broad all-features preset.

A direct configuration without a preset defaults the project feature options to
`ON`. Disable targets explicitly when configuring a narrower build. Presets and
CI commands keep their declared cache values and therefore override these
defaults.

When a build and test run is requested, the GCC debug path is:

```bash
cmake --preset tbb-gcc-debug
cmake --build build/tbb-gcc-debug
ctest --preset ctest-tbb-gcc-debug
```

To configure the engine alone without presets:

```bash
cmake -S . -B build/tbb-gcc -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_HOST_COMPILER=native \
    -DBUILD_TESTING=OFF -DATLAS_GOOGLE_TEST=OFF \
    -DATLAS_PYTHON=OFF -DATLAS_BENCHMARKS=OFF -DATLAS_EXAMPLES=OFF \
    -DATLAS_INTERACTIVE=OFF -DATLAS_INTERACTIVE_RENDERING=OFF
cmake --build build/tbb-gcc
```

Use a fresh build directory when switching compilers or backends. For a CUDA
build, use the `cuda-*` presets. Set `CMAKE_CUDA_ARCHITECTURES` explicitly when
building for another GPU; without it, configuration detects the local GPU or
falls back to `89-real`. Docker runtime builds provide their own explicit list.

---

## 3. Options and Constraints

Important options:

- `ATLAS_DEVICE_SYSTEM` — `TBB` (CPU, default) or `CUDA` (GPU)
- `ATLAS_HOST_COMPILER` — `native` (default) or `nvcc` for TBB
- `ATLAS_LOGGING`, `ATLAS_PYTHON`
- `ATLAS_GOOGLE_TEST`, `ATLAS_BENCHMARKS`, `ATLAS_EXAMPLES`
- `ATLAS_INTERACTIVE` — native execution/control library
- `ATLAS_INTERACTIVE_RENDERING` — OpenGL renderer and native executable when
  `ATLAS_INTERACTIVE` is enabled

Constraints:

- The Thrust host system is always TBB, and TBB is required in every
  configuration. The CUDA toolkit is required only when nvcc compiles the
  sources (`ATLAS_DEVICE_SYSTEM=CUDA` or `ATLAS_HOST_COMPILER=nvcc`).
- Benchmarks build under both backends; they are wired only when
  `ATLAS_BENCHMARKS` is on.
- `ATLAS_PYTHON` requires a Python 3.8+ interpreter with development headers and
  the nanobind submodule; see [python.md](python.md) for the module.
- `ATLAS_INTERACTIVE=ON` with `ATLAS_INTERACTIVE_RENDERING=OFF` requires no
  OpenGL stack. Rendering requires OpenGL, GLEW, GLFW, and GLM. See
  [interactive.md](interactive.md) for commands and target names.
- `ATLAS_INTERACTIVE_RENDERING` is a dependent option. It resolves to `OFF`
  whenever `ATLAS_INTERACTIVE` is disabled, avoiding a misleading graphics-on,
  execution-off cache configuration.

---

## 4. Test Authoring

- C++ tests use GoogleTest only. Include `<gtest/gtest.h>` directly.
- Test sources use the host compiler for native TBB and nvcc when
  `ATLAS_USE_NVCC` is enabled; they may exercise host/device-annotated APIs directly.
- Place tests under `tests/atlas/<module>/`, mirroring `include/atlas/` and
  `src/atlas/`.
- Name C++ test files `<subject>_tests.cpp`.
- Prefer self-contained tests that include only the headers under test. Keep
  local aliases, helper functions, and fixtures in an anonymous namespace.
- Prefer focused `TEST(SuiteName, BehaviorName)` cases that describe observable
  behavior.
- Do not add another test `main`; GoogleTest provides the entry point.

CMake globs `tests/atlas/**/*.cpp`, so a new file needs no CMake edit. Every
test lands in the aggregate `atlas_tests` target, and each directory also gets
its own executable named after the path relative to `tests/atlas/` — so
`tests/atlas/sink/` builds `atlas_tests_sink`.

Python binding tests live under `tests/python/` and use the standard-library
`unittest` runner against an installed wheel. See
[`tests/python/README.md`](../../tests/python/README.md) for the command and
coverage. They focus on the public object hierarchy, owned NumPy snapshots,
state transfers, native ownership across garbage collection, serialization,
controlled exceptions, and small deterministic simulation workflows. Backend
selection and TBB/CUDA parity cases use separate Python processes because a
loaded native engine cannot be replaced safely. These tests are separate from
the C++ GoogleTest targets.

Interactive C++ tests live under `tests/interactive/`. The execution, JSON,
Session, and Server cases link the headless `atlas::interactive` target and run
with `ATLAS_INTERACTIVE_RENDERING=OFF`. When rendering is enabled, a second
`atlas_tests_interactive_rendering` target verifies geometry mesh dispatch
without requiring a visible window. CMake registers the cases under the
corresponding target-name prefix.

Tests mirror `include/atlas/`: every module has a directory under
`tests/atlas/`. The covered modules are `buffer`, `codec`, `collider`,
`container`, `core`, `fluid`, `generator`, `geometry`, `logging`, `material`,
`math`, `memory`, `parallel`, `random`, `sampling`, `scan`,
`searcher`, `serialization`, `sink`, `solver`, `source`, `spatial`, `sync`,
`system`, `unit`, and `universe`.

## 5. GitHub CI

`.github/workflows/tbb.yml` runs only for manual dispatches on `main`.
It builds the aggregate `atlas_tests` target with GCC/G++ and TBB on Ubuntu
22.04 and 24.04, then runs CTest with `-R '^atlas_tests\.'`. This filter selects the
aggregate suite and avoids trying to execute the unbuilt per-module binaries.

`.github/workflows/python.yml` is a separate manual workflow on `main`, using
Ubuntu 22.04/Python 3.10 and Ubuntu 24.04/Python 3.12. It builds a TBB wheel
from the source archive, installs it, checks metadata and engine selection,
runs all `tests/python` tests, and executes the DSMC example.
Neither workflow exercises the CUDA backend. Artifact details and manual
PyPI publication are described in [releases.md](releases.md).

`.github/workflows/publish-docker.yml` builds the TBB and CUDA runtime images
for Ubuntu 22.04 and 24.04 on `linux/amd64`. Before optional GHCR publication,
it checks installed Python dependencies, loads the selected native engine,
checks basic math operations, and runs the DSMC example for TBB. The CUDA
checks run without a GPU and do not validate GPU simulation.
