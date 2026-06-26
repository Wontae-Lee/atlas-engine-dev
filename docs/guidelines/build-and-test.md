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

## 2. CMake Presets

`CMakePresets.json` requires CMake 3.20+ and Ninja. Important presets:

- Configure: `tbb-debug`, `tbb-release`, `cuda-debug`, `cuda-release`,
  `cuda-debug-tests`
- Build: `build-tbb-debug`, `build-tbb-release`, `build-cuda-debug`,
  `build-cuda-release`, `build-cuda-debug-tests`
- Test: `ctest-tbb-debug`, `ctest-tbb-release`

---

## 3. Options and Constraints

Important options:

- `ATLAS_USE_CUDA`, `ATLAS_USE_TBB`
- `ATLAS_USE_VIZKIT`, `ATLAS_LOGGING`
- `ATLAS_GOOGLE_TEST`, `ATLAS_CUDA_TEST`, `ATLAS_BENCHMARKS`

Constraints:

- Exactly one of `ATLAS_USE_CUDA` and `ATLAS_USE_TBB` must be enabled.
- `ATLAS_GOOGLE_TEST` is disabled for CUDA presets.
- Benchmarks are currently TBB-only.

---

## 4. Test Authoring

- Shared include: `src/testkit/testkit.h`. Include `<testkit/testkit.h>` instead
  of GoogleTest or cudatest headers directly.
- C++ tests use GoogleTest through `testkit`; CUDA tests use the in-tree
  `cudatest` through `testkit`.
- Place tests under `tests/<module>/` to match the public module or runtime
  subsystem.
- Name C++ test files `<subject>_tests.cpp` and CUDA companion files
  `<subject>_tests.cu`.
- Keep shared helpers in `tests/utilities/test_utils.h`; keep local aliases,
  helper functions, and fixtures in an anonymous namespace.
- Prefer focused `TEST(SuiteName, BehaviorName)` cases that describe observable
  behavior.
- Do not add another C++ or CUDA test `main`; entry points are provided by
  `testkit` and `tests/cuda/main.cu`.

Key test locations:

- `tests/system/system_tests.cpp`
- `tests/source/`
- `tests/sink/`
- `tests/observer/`
- `tests/material/`
- `tests/solver/`
