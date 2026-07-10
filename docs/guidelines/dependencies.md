# Dependencies and Reference Code

The external and in-tree dependencies, and the benchmark reference submodules.

---

## 1. Dependencies

**TBB** is required in every configuration; it backs the host-side parallel
algorithms (`parallel_for`, `parallel_sort`, `parallel_fill`).

The **CUDA 12.x toolkit** is required only when nvcc compiles the sources —
that is, `ATLAS_DEVICE_SYSTEM=CUDA`, or `ATLAS_HOST_COMPILER=nvcc`. A default
`ATLAS_DEVICE_SYSTEM=TBB` build needs neither nvcc nor Thrust: the buffers are
`std::vector` and the algorithms are TBB's. A GPU is needed only to *run*
`ATLAS_DEVICE_SYSTEM=CUDA` builds.

CMake defines exactly one of `ATLAS_BACKEND_CUDA` and `ATLAS_BACKEND_TBB`. Ten
headers branch on it to pick the container and the algorithm — nine under
`buffer/`, `memory/`, `parallel/`, and `scan/`, plus `geometry/triangle_mesh.h`
(where the host cannot dereference a device-resident BVH). A few other files
mention Thrust only in documentation comments.

The reference development environment is the Docker `dev` image defined in
`Dockerfile`. Building the optional Python bindings additionally requires a
Python 3.8+ interpreter with the development headers.

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

`benchmarks/` currently holds only Atlas's own benchmark targets — the
`benchmarks/atlas/` cases (for example `cylinder/`), built on Google Benchmark
and wired only when `ATLAS_BENCHMARKS` is on.

The four external reference projects — `piclas`, `sparta`, `splishsplash`, and
`dumux` — were once wired through `ExternalProject_Add`, but they were removed
with the engine restructuring and were never registered in `.gitmodules`; see
the note in [`benchmarks/CMakeLists.txt`](../../benchmarks/CMakeLists.txt). They
are expected to be reinstated later. When the user mentions one of these names in
a benchmark or reference-code context, treat it as one of those reference
projects rather than a directory that currently exists in the tree.
