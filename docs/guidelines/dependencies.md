# Dependencies and Reference Code

The external and in-tree dependencies, and the benchmark reference submodules.

---

## 1. Dependencies

External dependencies are TBB and the CUDA 12.x toolkit — both are required in
every configuration: Atlas compiles exclusively with nvcc, TBB is the Thrust
host system (and the CPU device system), and Thrust ships with the toolkit.
A GPU is needed only to run `ATLAS_DEVICE_SYSTEM=CUDA` builds. The reference
development environment is the Docker `dev` image defined in `Dockerfile`.
Building the optional Python bindings additionally requires a Python 3.8+
interpreter with the development headers.

In-tree dependencies under `external/` include:

- tinyobjloader
- Lyra
- googletest
- googlebenchmark
- protobuf
- nanobind (Python bindings)

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
