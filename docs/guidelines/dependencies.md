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
headers under `buffer/`, `memory/`, `parallel/`, and `scan/` branch on it to pick
the container and the algorithm. Nothing else in the tree names Thrust.

The reference development environment is the Docker `dev` image defined in
`Dockerfile`. Building the optional Python bindings additionally requires a
Python 3.8+ interpreter with the development headers.

In-tree dependencies under `external/` include:

- tinyobj (tinyobjloader)
- googletest
- googlebenchmark
- protobuf
- nanobind (Python bindings; carries its own submodule, so a clone needs
  `git submodule update --init --recursive external/nanobind`)

Do not introduce new dependencies unless explicitly requested.

---

## 2. Benchmark Reference Submodules

Benchmark reference submodules live under `benchmarks/`:

- `benchmarks/dumux`
- `benchmarks/piclas`
- `benchmarks/sparta`
- `benchmarks/splishsplash`

When the user mentions `piclas`, `dumux`, `sparta`, or `splishsplash` in a
benchmark or reference-code context, treat the name as referring to the
corresponding submodule directory.
