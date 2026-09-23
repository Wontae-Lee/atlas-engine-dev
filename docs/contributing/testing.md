# Testing Atlas

Test policy, suite layout, and change-specific validation.

## Run policy

Do not run builds, tests, benchmarks, simulations, generators, or formatters
unless the user explicitly asks.

The Python bindings are packaged with scikit-build-core; when the user asks to
build them, use `scripts/build_wheels.sh` (it installs its own toolchain via
`pip`). The repo provisions no `.venv`.

---

## Test authoring

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


## Validation by change type

- Core algorithm or state changes: build and run the relevant C++ module tests on TBB; compile CUDA and run it when a CUDA-capable environment is available.
- Python binding changes: build the selected native extension and run `tests/python/` against the installed package. Backend parity requires separate processes.
- Interactive execution changes: run the headless Interactive tests. Rendering changes also build the rendering target and its geometry tests.
- Example changes: build the corresponding executable or install the Python package, then run the maintained cylinder smoke case.
- Benchmark harness changes: build and run the small benchmark smoke configuration; performance claims require a separate deliberate benchmark run.

Configured CI coverage is not evidence that a particular local or hosted run passed. Report only commands that actually completed.
