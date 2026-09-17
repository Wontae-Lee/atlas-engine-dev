# Dependencies and Reference Code

The external and in-tree dependencies, and the benchmark reference submodules.

---

## 1. Dependencies

**TBB** is required in every configuration; it backs the host-side parallel
algorithms (`parallel_for`, `parallel_sort`, `parallel_fill`).

Native TBB builds require a C compiler and a C++20 compiler, CMake 3.20+, and
Ninja for the provided presets (whose schema requires CMake 3.21+).
`tbb-gcc-debug` and `tbb-gcc-release` select
`gcc` and `g++` explicitly; `tbb-debug` and `tbb-release` use CMake's selected
host toolchain. See [build-and-test.md](build-and-test.md) for the preset and
direct configure commands.

The **CUDA 12.x toolkit** is required only when nvcc compiles the sources —
that is, `ATLAS_DEVICE_SYSTEM=CUDA`, or `ATLAS_HOST_COMPILER=nvcc`. A default
`ATLAS_DEVICE_SYSTEM=TBB` / `ATLAS_HOST_COMPILER=native` build needs neither nvcc
nor Thrust: the buffers are `std::vector` and the algorithms are TBB's. CUDA is
not enabled or searched for in this configuration. A GPU is needed only to
*run* `ATLAS_DEVICE_SYSTEM=CUDA` builds. Even when nvcc is enabled, host-only
serialization, logging, and external dependencies still use the C/C++ compilers.

CMake defines exactly one of `ATLAS_BACKEND_CUDA` and `ATLAS_BACKEND_TBB`. Ten
headers branch on it to pick the container and the algorithm — nine under
`buffer/`, `memory/`, `parallel/`, and `scan/`, plus `geometry/triangle_mesh.h`
(where the host cannot dereference a device-resident BVH). A few other files
mention Thrust only in documentation comments.

The Docker `tbb-dev` image provides a native GCC/G++ toolchain without CUDA;
`cuda-dev` adds nvcc, and `dev` remains its compatibility alias. Both include
Python development headers and a virtual environment. The separate `tbb` and
`cuda` runtime images include the installed Atlas Python package, NumPy, and
their runtime libraries. Their default base is Ubuntu 22.04; Ubuntu 24.04 is
selected with `--build-arg UBUNTU_VERSION=24.04`. See [docker.md](docker.md).
Building the optional Python bindings outside these images requires a
Python 3.8+ interpreter with development headers.

Dependencies under `external/` are git submodules, not vendored copies; a fresh
clone needs `git submodule update --init --recursive` to populate them:

- tinyobj (tinyobjloader)
- googletest
- googlebenchmark
- protobuf
- nanobind (Python bindings; carries nested submodules, so its init must be
  `--recursive`)

Do not introduce new dependencies unless explicitly requested.

---

## 2. Benchmark Reference Code

`benchmarks/` holds Atlas's own Google Benchmark cases under `benchmarks/atlas/`
(each a `main.cpp` built into `atlas_benchmark_<case>_gbench`), wired only when
`ATLAS_BENCHMARKS` is on. Only a `smoke` case exists for now; the representative
cases are to be rewritten. Standalone, framework-free simulations live under
[`examples/cpp/`](../../examples/cpp/) instead (for example `cylinder/`), built
with `ATLAS_EXAMPLES`.

The four external reference projects — `piclas`, `sparta`, `splishsplash`, and
`dumux` — were once wired through `ExternalProject_Add`, but they were removed
with the engine restructuring and were never registered in `.gitmodules`; see
the note in [`benchmarks/CMakeLists.txt`](../../benchmarks/CMakeLists.txt). They
are expected to be reinstated later. When the user mentions one of these names in
a benchmark or reference-code context, treat it as one of those reference
projects rather than a directory that currently exists in the tree.
