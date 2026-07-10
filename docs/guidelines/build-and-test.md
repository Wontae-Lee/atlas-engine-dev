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

Atlas is compiled exclusively with **nvcc**: every translation unit (engine
`.cu` sources, tests, benchmarks) is compiled as CUDA, and the backend is the
Thrust device system selected at configure time. A CUDA toolkit is therefore
always required to build — a GPU is required only to run the CUDA variant.

The reference development environment is the Docker `dev` image
(see `Dockerfile`):

```bash
docker build --target dev -t atlas-dev .
docker run --rm -it -v "$PWD":/workspace atlas-dev
```

`CMakePresets.json` requires CMake 3.20+ and Ninja. Important presets:

- Configure: `tbb-debug`, `tbb-release` (CPU; Thrust device = TBB),
  `cuda-debug`, `cuda-release` (GPU; Thrust device = CUDA)
- Build: `build-tbb-debug`, `build-tbb-release`, `build-cuda-debug`,
  `build-cuda-release`
- Test: `ctest-tbb-debug`, `ctest-tbb-release`

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

Current test locations:

- `tests/atlas/collider/`
- `tests/atlas/generator/`
- `tests/atlas/material/`
- `tests/atlas/sink/`
- `tests/atlas/source/`
