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

`ATLAS_DEVICE_SYSTEM` selects the parallel backend. `CUDA` compiles the engine's
`.cu` sources and the enabled Atlas tests, examples, benchmarks, and Python
bindings with nvcc, and uses Thrust's containers and algorithms. Host-only
serialization, logging, and external dependencies keep their C/C++ compilers.
`TBB` (the default) uses `std::vector` buffers and TBB algorithms, without Thrust.

`ATLAS_HOST_COMPILER` selects the compiler mode for TBB. `native` (the default)
uses `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`, such as `gcc` and `g++`, without
enabling CUDA or finding the CUDA toolkit. CMake marks the engine and example
`.cu` sources as C++ in this mode. They hold no CUDA-only syntax — every kernel
is a `parallel_for` over an `ATLAS_ALL_DEVICE` lambda, and those annotations
vanish outside `__CUDACC__`. `nvcc` compiles those sources and the enabled Atlas
tests and benchmarks through nvcc while retaining the TBB backend. Host-only
code still uses the C/C++ compilers; the TBB Python bindings also keep the host
C++ compiler.

Keep a CI job on `tbb-nvcc-debug`: **nvcc rejects constructs the host compiler
accepts** — notably an extended `__host__ __device__` lambda inside a private
member function, which is the only reason `System::mark_survivors` and friends
are public. A TBB-only build stops catching those.

A CUDA toolkit is therefore needed only when nvcc is in play. A GPU is needed
only to *run* the CUDA variant.

The reference development environment is the Docker `dev` image
(see `Dockerfile`):

```bash
docker build --target dev -t atlas-dev .
docker run --rm -it -v "$PWD":/workspace atlas-dev
```

`CMakePresets.json` requires CMake 3.20+ and Ninja. Important presets:

- Configure: `tbb-debug`, `tbb-release` (CPU; host compiler, no CUDA toolkit),
  `tbb-gcc-debug`, `tbb-gcc-release` (CPU; explicitly `gcc`/`g++`),
  `tbb-nvcc-debug` (CPU backend built by nvcc),
  `cuda-debug`, `cuda-release` (GPU; nvcc + Thrust)
- Build: `build-tbb-debug`, `build-tbb-release`,
  `build-tbb-gcc-debug`, `build-tbb-gcc-release`, `build-tbb-nvcc-debug`,
  `build-cuda-debug`, `build-cuda-release`
- Test: `ctest-tbb-debug`, `ctest-tbb-gcc-debug`, `ctest-tbb-nvcc-debug`,
  `ctest-cuda-debug`

The debug presets turn logging, tests, the Python module, the examples, and the
benchmarks on; the release presets turn all five off.

To select GCC explicitly with a preset:

```bash
cmake --preset tbb-gcc-debug
cmake --build --preset build-tbb-gcc-debug
ctest --preset ctest-tbb-gcc-debug
```

Without presets, select the backend and compilers directly:

```bash
cmake -S . -B build/tbb-gcc -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_HOST_COMPILER=native \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/tbb-gcc
```

`native` is a compiler mode, not another name for GCC: the `tbb-debug` and
`tbb-release` presets leave compiler selection to CMake, a toolchain file, or
`CC`/`CXX` on the first configure. The `tbb-gcc-*` presets set both compiler
variables explicitly. Use a fresh build directory when changing compilers or
switching between native and nvcc modes; CMake caches the toolchain during the
first configure. Each preset already has its own `build/<preset>` directory.

---

## 3. Options and Constraints

Important options:

- `ATLAS_DEVICE_SYSTEM` — `TBB` (CPU, default) or `CUDA` (GPU)
- `ATLAS_HOST_COMPILER` — `native` (default) or `nvcc` for TBB builds
- `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` — host C/C++ compiler executables;
  the GCC presets select `gcc` and `g++`
- `ATLAS_LOGGING`, `ATLAS_PYTHON`
- `ATLAS_GOOGLE_TEST`, `ATLAS_BENCHMARKS`, `ATLAS_EXAMPLES`

Constraints:

- TBB is required in every configuration and is also the Thrust host system
  in CUDA builds. The CUDA toolkit is required only when nvcc compiles the
  sources (`ATLAS_DEVICE_SYSTEM=CUDA` or `ATLAS_HOST_COMPILER=nvcc`).
- Benchmarks build under both backends; they are wired only when
  `ATLAS_BENCHMARKS` is on.
- `ATLAS_PYTHON` requires a Python 3.8+ interpreter with development headers and
  the nanobind submodule; see [python.md](python.md) for the module.

---

## 4. Test Authoring

- Tests use GoogleTest only. Include `<gtest/gtest.h>` directly.
- C++ test sources use the native C++ compiler in native TBB builds and nvcc
  (marked as CUDA in CMake) when `ATLAS_USE_NVCC` is enabled. Keep tests portable
  across both modes; they may exercise host/device-annotated APIs directly.
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

Tests mirror `include/atlas/`: every module has a directory under
`tests/atlas/`. The covered modules are `buffer`, `codec`, `collider`,
`container`, `core`, `fluid`, `generator`, `geometry`, `logging`, `material`,
`math`, `memory`, `observer`, `parallel`, `random`, `sampling`, `scan`,
`searcher`, `serialization`, `sink`, `solver`, `source`, `spatial`, `sync`,
`system`, `unit`, and `universe`.

## 5. GitHub TBB CI

`.github/workflows/tbb.yml` runs only for manual dispatches on `main`.
It builds the aggregate `atlas_tests` target with the native compiler
and TBB, then runs CTest with `-R '^atlas_tests\.'`. This filter selects the
aggregate suite and avoids trying to execute the unbuilt per-module binaries.
Python packaging checks build a wheel from the source archive, install it, and
run the existing DSMC example. Release packaging and manual PyPI publication are
described in [python.md](python.md#github-ci-and-publication).
