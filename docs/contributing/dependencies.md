# Dependencies

External and in-tree dependencies used by Atlas builds.

**TBB** is required in every configuration; it backs the host-side parallel
algorithms (`parallel_for`, `parallel_sort`, `parallel_fill`).

Native TBB builds require a C compiler and a C++20 compiler, CMake 3.20+, and
Ninja for the provided presets (whose schema requires CMake 3.21+).
`tbb-gcc-debug` and `tbb-gcc-release` select
`gcc` and `g++` explicitly; `tbb-debug` and `tbb-release` use CMake's selected
host toolchain. See [build.md](build.md) for the preset and
direct configure commands.

The **CUDA toolkit** is required only when nvcc compiles the sources —
that is, `ATLAS_DEVICE_SYSTEM=CUDA`, or `ATLAS_HOST_COMPILER=nvcc`. A default
`ATLAS_DEVICE_SYSTEM=TBB` / `ATLAS_HOST_COMPILER=native` build needs neither nvcc
nor Thrust: the buffers are `std::vector` and the algorithms are TBB's. CUDA is
not enabled or searched for in this configuration. A GPU is needed only to
*run* `ATLAS_DEVICE_SYSTEM=CUDA` builds. Even when nvcc is enabled, host-only
serialization, logging, and external dependencies still use the C/C++ compilers.
CUDA 13 moves CCCL, including Thrust, below the toolkit's `include/cccl`
directory. CMake propagates that directory when present so host-compiled Atlas
translation units see the same headers as nvcc; CUDA 12's direct include layout
continues to work.

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
selected with `--build-arg UBUNTU_VERSION=24.04`. See [Docker operations](../operations/docker.md).
The same `tbb` and `cuda` images include OpenGL runtime libraries and the native
Atlas executables. Python remains the default command; native applications are
invoked explicitly.
Building the optional Python bindings outside these images requires a
Python 3.8+ interpreter with development headers.

The interactive execution/control target adds no graphics dependency. With
`ATLAS_INTERACTIVE_RENDERING=ON`, the native renderer additionally requires
OpenGL, GLEW, GLFW, and GLM. CUDA interactive builds also use the CUDA/OpenGL
interop API. See [Interactive frontend](../frontends/interactive.md) for the headless and rendering
configure commands. The execution/control target uses the header-only
**nlohmann/json** submodule for JSON configuration and JSONL protocol messages.

Dependencies under `external/` are git submodules, not vendored copies; a fresh
clone needs `git submodule update --init --recursive` to populate them:

- tinyobj (tinyobjloader)
- googletest
- googlebenchmark
- protobuf
- nlohmann/json (interactive configuration and control protocol)
- nanobind (Python bindings; carries nested submodules, so its init must be
  `--recursive`)

Do not introduce new dependencies unless explicitly requested.
