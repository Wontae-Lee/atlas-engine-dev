# Build and Test

The build/test policy, the CMake presets and options, and the rules for
authoring tests.

---

## 1. Run Policy

Do not run builds, tests, benchmarks, simulations, generators, or formatters
unless the user explicitly asks.

If the user asks to run project Python tools, activate the virtual environment
first:

```bash
source .venv/bin/activate
```

---

## 2. Toolchain and CMake Presets

Two switches pick the toolchain.

`ATLAS_DEVICE_SYSTEM` selects the parallel backend. `CUDA` compiles every
translation unit with nvcc and uses Thrust's containers and algorithms. `TBB`
(the default) uses neither: buffers are `std::vector`, the algorithms are TBB's,
and the host compiler builds the whole tree. The `.cu` suffix survives on the
engine sources because they hold no CUDA-only syntax — every kernel is a
`parallel_for` over an `ATLAS_ALL_DEVICE` lambda, and those annotations vanish
outside `__CUDACC__`.

`ATLAS_HOST_COMPILER` (`native` | `nvcc`) forces a TBB build through nvcc.
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
  `tbb-nvcc-debug` (CPU backend built by nvcc),
  `cuda-debug`, `cuda-release` (GPU; nvcc + Thrust)
- Build: `build-tbb-debug`, `build-tbb-release`, `build-tbb-nvcc-debug`,
  `build-cuda-debug`, `build-cuda-release`
- Test: `ctest-tbb-debug`, `ctest-tbb-nvcc-debug`, `ctest-cuda-debug`

The debug presets turn logging, tests, the Python module, and the benchmarks on;
the release presets turn all four off.

---

## 3. Options and Constraints

Important options:

- `ATLAS_DEVICE_SYSTEM` — `TBB` (CPU, default) or `CUDA` (GPU)
- `ATLAS_LOGGING`, `ATLAS_PYTHON`
- `ATLAS_GOOGLE_TEST`, `ATLAS_BENCHMARKS`

Constraints:

- The Thrust host system is always TBB; TBB and the CUDA toolkit are required
  in every configuration.
- Benchmarks are currently TBB-only (disabled when `ATLAS_DEVICE_SYSTEM=CUDA`).
- `ATLAS_PYTHON` requires a Python 3.8+ interpreter with development headers.

---

## 4. Test Authoring

- Tests use GoogleTest only. Include `<gtest/gtest.h>` directly.
- Test sources are compiled by nvcc (marked as CUDA in CMake); they may
  exercise host/device-annotated APIs directly.
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

Tests mirror `include/atlas/`: nearly every module has a directory under
`tests/atlas/`. The covered modules are `buffer`, `codec`, `collider`,
`container`, `core`, `fluid`, `generator`, `geometry`, `material`, `math`,
`memory`, `observer`, `parallel`, `random`, `sampling`, `scan`, `searcher`,
`sink`, `solver`, `source`, `spatial`, `sync`, `system`, `unit`, and
`universe`.
