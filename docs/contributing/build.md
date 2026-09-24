# Building Atlas

CMake configuration, toolchain selection, and target boundaries.

## Toolchain and CMake presets

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

## Standard Docker development

Build the development image once, then mount the checkout through the Python
launcher. Commands later in this guide run inside that development shell:

```bash
python3 scripts/dev.py build tbb
python3 scripts/dev.py tbb
```

For nvcc and host GPU access:

```bash
python3 scripts/dev.py build cuda
python3 scripts/dev.py cuda
```

The launcher also accepts one command, for example
`python3 scripts/dev.py tbb cmake --preset tbb-gcc-debug`. It preserves build
outputs and caches separately for each backend and Ubuntu version. Use fresh
build directories when changing compilers or toolchain images. CLion can use the
same image as a Docker toolchain with `/workspace` as the source path; select a
matching CMake preset. Host-only workflows can install the dependencies with
[the dependency installer](dependencies.md#installation-outside-docker).

`ATLAS_DOCKER_GPU=0` allows a CUDA compilation container on a host without GPU
access. Running CUDA tests still requires a GPU. TBB images contain no CUDA
toolkit. See [Docker operations](../operations/docker.md) for image versions,
cache paths, display forwarding, and runtime images.

The project requires CMake 3.20+; the version-3 `CMakePresets.json` format
requires CMake 3.21+. The presets use Ninja. Important presets:

- Configure: `tbb-debug`, `tbb-release` (CPU; host compiler, no CUDA toolkit),
  `tbb-gcc-debug`, `tbb-gcc-release` (explicit `gcc`/`g++`),
  `tbb-nvcc-debug` (CPU backend built by nvcc),
  `cuda-debug`, `cuda-release` (GPU; nvcc + Thrust),
  `tbb-application-release`, `cuda-application-release` (native applications)
- Complete native validation: configure/build/test presets `tbb-test` and
  `cuda-test` enable Core and Interactive tests, rendering, examples, and benchmarks.
  Their CTest filters run the aggregate Core suite once and both Interactive suites.
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

A typical GCC debug configuration and test run is:

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

## Options and constraints

Important options:

- `ATLAS_DEVICE_SYSTEM` — `TBB` (CPU, default) or `CUDA` (GPU)
- `ATLAS_HOST_COMPILER` — `native` (default) or `nvcc` for TBB
- `ATLAS_LOGGING`, `ATLAS_PYTHON`
- `ATLAS_GOOGLE_TEST`, `ATLAS_BENCHMARKS`, `ATLAS_EXAMPLES`
- `ATLAS_INTERACTIVE` — native execution/control library
- `ATLAS_INTERACTIVE_RENDERING` — OpenGL renderer linked into the native
  executable when `ATLAS_INTERACTIVE` is enabled; the executable also exists in
  headless builds

Constraints:

- The Thrust host system is always TBB, and TBB is required in every
  configuration. The CUDA toolkit is required only when nvcc compiles the
  sources (`ATLAS_DEVICE_SYSTEM=CUDA` or `ATLAS_HOST_COMPILER=nvcc`).
- Benchmarks build under both backends; they are wired only when
  `ATLAS_BENCHMARKS` is on.
- `ATLAS_PYTHON` requires a Python 3.8+ interpreter with development headers and
  the pinned nanobind Python package; see the [Python frontend](../frontends/python.md).
- `ATLAS_INTERACTIVE=ON` with `ATLAS_INTERACTIVE_RENDERING=OFF` requires no
  OpenGL stack. Rendering requires OpenGL, GLEW, GLFW, and GLM. See
  [Interactive frontend](../frontends/interactive.md) for commands and target
  names.
- `ATLAS_INTERACTIVE_RENDERING` is a dependent option. It resolves to `OFF`
  whenever `ATLAS_INTERACTIVE` is disabled, avoiding a misleading graphics-on,
  execution-off cache configuration.

---
