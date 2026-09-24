# Dependencies

Docker is the standard development environment. `tbb-dev` and `cuda-dev`
install the same pinned host libraries under `/opt/atlas-deps`; `cuda-dev` adds
nvcc. Atlas CMake uses installed package targets through `find_package()` and
never downloads or builds third-party sources during an Atlas configure.
There are no Git submodules to initialize.

## Versions and installation

[`cmake/dependencies/CMakeLists.txt`](../../cmake/dependencies/CMakeLists.txt)
records release archive URLs and SHA-256 hashes. The separate dependency build
installs static, position-independent C++ libraries with C++20 and the same host
toolchain used by Atlas. Protobuf's runtime and `protoc` come from one release.
Its Abseil version follows that release's upstream dependency declaration;
`protobuf_LOCAL_DEPENDENCIES_ONLY` prevents an unpinned fallback download.

| Dependency | Version | Consumer |
|---|---|---|
| Protobuf | 36.2 | Core snapshot serialization and `protoc` |
| Abseil | 20250512.1 | Protobuf |
| tinyobjloader | 2.0.0rc13 | Core mesh loading |
| GoogleTest | 1.18.0 | C++ tests |
| Google Benchmark | 1.9.5 | Benchmarks |
| nlohmann/json | 3.12.0 | Interactive configuration and JSONL |
| nanobind | 2.13.0 | Python extension builds |
| oneTBB | Ubuntu package; 2021.11.0 in manylinux | Host parallel algorithms |

nanobind is installed from PyPI, with the same version in `pyproject.toml` and
[`docker/requirements-build.txt`](../../docker/requirements-build.txt). CMake
locates it through the selected Python interpreter. Installing nanobind provides
its source and CMake helpers; the extension still compiles its binding code and
nanobind runtime. The Core and Interactive targets do not depend on nanobind.

Ubuntu supplies the compilers, TBB, Python development headers, and OpenGL,
GLEW, GLFW, and GLM packages listed in `docker/packages.txt`. OS packages follow
the selected Ubuntu repository; release archive pins do not freeze apt packages
or transitive Python packages. Use an image digest when reusing an exact image.
Third-party license notices are retained under `licenses/third-party/`, included
in wheels and source distributions, and copied into runtime images.

## Installation outside Docker

Native host builds remain supported with installed dependencies. On Ubuntu,
install the packages in `docker/packages.txt`, create a Python virtual
environment if building bindings, then run:

```bash
python -m pip install -r docker/requirements-build.txt
ATLAS_DEPS_PREFIX="$PWD/build/dependencies-install" python3 scripts/install_dependencies.py
export CMAKE_PREFIX_PATH="$PWD/build/dependencies-install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
cmake --preset tbb-gcc-debug
```

The first dependency installation needs network access to download the verified
archives. Later Atlas builds need only the installed packages. To choose another
build or install directory, set `ATLAS_DEPS_BUILD_DIR` or `ATLAS_DEPS_PREFIX`.
Pass `-DATLAS_DEPS_DEVELOPMENT=OFF` for only the Core/Python C++ dependencies.
Pass `-DATLAS_DEPS_WITH_TBB=ON` when the OS does not supply a suitable oneTBB.
The installer invokes independent CMake projects sequentially; each compilation
uses `CMAKE_BUILD_PARALLEL_LEVEL` (default 2).

Source distributions contain this installer and the pinned dependency manifest,
not third-party source trees. Building a source distribution outside the standard
image requires installing these native dependencies first. The manylinux wheel
workflow runs the same installer with oneTBB enabled before building wheels.
For oneTBB 2021.11, the installer sets the CMake compatibility policy minimum to
3.5 so the release can also be configured by CMake 4.

## Backend and frontend boundaries

TBB is required for host-side parallel algorithms in both backends. A native TBB
build requires a C/C++20 toolchain, CMake 3.20+ (3.21+ for presets), and Ninja;
it does not search for CUDA or require Thrust. CUDA is enabled only for
`ATLAS_DEVICE_SYSTEM=CUDA` or `ATLAS_HOST_COMPILER=nvcc`. A GPU is needed to run
CUDA code, not to compile it. Protobuf and other host libraries use the C/C++
compiler even in CUDA configurations.

CMake defines exactly one of `ATLAS_BACKEND_TBB` and `ATLAS_BACKEND_CUDA`.
CUDA 13's `include/cccl` layout and CUDA 12's direct include layout are both
supported. Backend-specific containers and algorithms stay behind Core's
backend abstraction.

Interactive execution uses JSON without graphics dependencies. OpenGL, GLEW,
GLFW, and GLM are found only with `ATLAS_INTERACTIVE_RENDERING=ON`; CUDA rendering
also uses CUDA/OpenGL interop. The Python extension requires Python 3.8+
development headers. See [build](build.md), [Docker](../operations/docker.md),
and [Python](../frontends/python.md) for commands.
