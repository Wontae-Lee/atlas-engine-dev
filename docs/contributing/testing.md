# Testing Atlas

Test policy, suite layout, and change-specific validation.

## Packaging and local runs

The Python bindings are packaged with scikit-build-core. `scripts/build_wheels.py`
builds and repairs wheels using the packaging toolchain preinstalled in the
standard development image. The container provides `/opt/venv`.

## Complete validation in Docker

From the host, build an image and run its native suites:

```bash
python3 scripts/dev.py build tbb
python3 scripts/dev.py tbb cmake --preset tbb-test
python3 scripts/dev.py tbb cmake --build --preset tbb-test
python3 scripts/dev.py tbb ctest --preset tbb-test
python3 scripts/dev.py tbb python scripts/check_python_package.py
```

The native build includes all Core cases, Interactive execution and rendering
geometry tests, native examples, and the benchmark smoke target. CTest filters
out duplicate per-directory Core executables. Rendering geometry tests need no
visible window. To also validate the graphics-independent configuration,
configure a separate directory with `ATLAS_INTERACTIVE_RENDERING=OFF`.

For GPU validation, use the `cuda` launcher and `cuda-test` presets; the launcher
passes `--gpus all`. The CUDA toolchain detects the host GPU architecture, or it
can be supplied with `-DCMAKE_CUDA_ARCHITECTURES=89-real` at configure time.
Run CUDA CTest serially to avoid competing GPU allocations.

For Python TBB/CUDA parity, enter one CUDA development shell and run:

```bash
python scripts/build_wheels.py
python -m pip install --force-reinstall --no-deps dist/cuda/atlas_engine-*.whl
ATLAS_DEFAULT_ENGINE=tbb python -X faulthandler -m unittest discover -s tests/python -v
ATLAS_DEFAULT_ENGINE=cuda python -X faulthandler -m unittest discover -s tests/python -v
```

The combined wheel provides both extensions, allowing the parity tests to run.
The package validation script separately builds a TBB wheel from the source
distribution, checks metadata and dependencies, runs all Python tests, and runs
the maintained example. It keeps artifacts under `dist/python-validation/`.

---

## Test authoring

- C++ tests use GoogleTest only. Include `<gtest/gtest.h>` directly.
- Test sources use the host compiler for native TBB and nvcc when
  `ATLAS_USE_NVCC` is enabled; they may exercise host/device-annotated APIs directly.
- Define extended host/device lambdas in namespace-scope helpers when nvcc builds
  a GoogleTest case: the generated `TEST` body is a private member function,
  which nvcc does not allow to enclose such a lambda.
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

The [CI guide](../operations/ci.md) lists configured checks. Check a particular
workflow run before interpreting its results as a passing validation.
