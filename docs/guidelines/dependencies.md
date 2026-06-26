# Dependencies and Reference Code

The external and in-tree dependencies, and the benchmark reference submodules.

---

## 1. Dependencies

External dependencies include TBB, CUDA 12.x, and the OpenGL stack used by
Vizkit.

In-tree dependencies under `external/` include:

- tinyobjloader
- Lyra
- googletest
- googlebenchmark
- protobuf

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
