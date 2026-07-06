# 4. Backend Model and Portability

Atlas is compiled with a single toolchain — **nvcc** — for every configuration.
The execution backend is selected through the **Thrust device system** at
configure time. This document describes how the backend is selected and how
code stays backend-agnostic.

---

## 4.1 One Toolchain, Two Device Systems

Every Atlas translation unit (engine sources, tests, benchmarks) is compiled as
CUDA by nvcc. The backend is chosen with one CMake option:

```text
ATLAS_DEVICE_SYSTEM = TBB      (CPU build; Thrust device system = TBB)
ATLAS_DEVICE_SYSTEM = CUDA     (GPU build; Thrust device system = CUDA)
```

The Thrust host system is **always TBB**. The resulting Thrust macros are:

```text
THRUST_HOST_SYSTEM   = THRUST_HOST_SYSTEM_TBB          (always)
THRUST_DEVICE_SYSTEM = THRUST_DEVICE_SYSTEM_TBB | THRUST_DEVICE_SYSTEM_CUDA
```

Key consequences:

- A GPU is required only to **run** the CUDA variant, never to build it.
- The TBB variant generates no GPU code, so it configures and compiles
  everywhere nvcc runs, and device-system algorithms execute on the CPU with
  TBB parallelism.
- Because nvcc compiles everything, `.cu` sources and host/device-annotated
  headers behave identically in both variants.

Compile-time backend branches key off these macros:

```text
ATLAS_TASKING_TBB     active when ATLAS_DEVICE_SYSTEM=TBB
ATLAS_TASKING_CUDA    active when ATLAS_DEVICE_SYSTEM=CUDA
```

---

## 4.2 Scalar Type

The engine uses a single scalar type: **`float`**. Runtime classes are plain
(non-template) types — `Vector3`, `Matrix3x3`, `Quaternion`, and so on — with
integer companions (`Vector3i`) where grid indexing needs them. Do not
introduce new `template <typename T>` scalar parameterization; the historical
template-based design is being removed module by module.

---

## 4.3 Shared Aliases

Backend differences in storage and ownership are hidden behind shared aliases.
Prefer these over backend-specific types:

| Alias | TBB device system | CUDA device system |
|---|---|---|
| `DeviceBuffer<T>` | CPU-resident storage | `thrust::device_vector<T>` |
| `HostBuffer<T>` | host `std::vector`-like storage | `thrust::host_vector<T>` |
| `device_shared_ptr<T>` | `std::shared_ptr<T>` | CUDA-aware managed ownership |
| `host_shared_ptr<T>` | `std::shared_ptr<T>` | `std::shared_ptr<T>` |

`DeviceBuffer<T>` holds data the active device system processes (on the GPU
under CUDA, on the CPU under TBB). `HostBuffer<T>` is host-resident
staging/config data.

---

## 4.4 Parallel Dispatch

Backend-agnostic parallelism goes through the `parallel_for` family and the
companion algorithms, parameterized by an execution policy:

```cpp
parallel_for<ExecutionPolicy::device>(begin, end, func);
```

`ExecutionPolicy` selects `serial`, `host`, or `device` execution; the
implementation dispatches at compile time to the active Thrust systems.
Companion primitives follow the same policy-parameterized shape:

```text
parallel_for, parallel_fill, parallel_sort   (parallel/)
exclusive_scan                               (scan/)
transform, transform_reduce                  (transform/)
remove_if                                     (remove/)
```

Use these instead of writing raw `thrust::` or `tbb::` calls in core code.

---

## 4.5 Header-Inline vs. Compiled Code

Atlas does not use CUDA relocatable device code (`-rdc`). A device-callable
function must therefore be visible (inline) in every translation unit that
launches kernels using it. This yields the placement rule:

```text
- Device-callable code (anything marked ATLAS_DEVICE / ATLAS_ALL_DEVICE)
  is defined inline in headers under include/atlas/.
- Host-only code (builders, validation, orchestration, I/O) is declared in
  headers and defined in src/atlas/<module>/*.cu.
```

The math module is entirely device-hot and is therefore header-inline by
design. Host-heavy modules split into `.h` declaration + `.cu` definition; see
`material/` for the reference example of the split.

---

## 4.6 Portability Macros

`include/atlas/core/macros.h` provides the host/device portability layer:

```text
ATLAS_HOST          mark host-callable code
ATLAS_DEVICE        mark device-callable code
ATLAS_ALL_DEVICE    mark code callable from host and device
ATLAS_FORCE_INLINE  aggressive inlining across compilers
ATLAS_NODISCARD     warn on ignored results
ATLAS_MAYBE_UNUSED  suppress unused-entity warnings
RESTRICT            restrict-qualified pointer hint
```

A function reachable from device kernels must carry the appropriate
host/device attributes; a header-inline hot-path function typically combines
an attribute with `ATLAS_FORCE_INLINE`.

---

## 4.7 Device-Callable Operators

Virtual interfaces (for example `Geometry`, `Generator`, `Solver`, `Searcher`)
are convenient on the host but cannot be called through vtables inside kernels.
The pattern Atlas uses is a **device-callable operator**: a value-type
union/variant that encodes the concrete behavior without virtual dispatch.

```text
Geometry      ->  GeometryOperator          (geometry/)
Generator     ->  Generate          (generator/)
surface model ->  SurfaceInteractionKernel  (collider/interaction/)
```

Host code builds the operator from a polymorphic description; kernels receive
the operator by value and dispatch on its tag. When adding a new geometry,
generator, or interaction model, extend the matching operator union so the new
case is reachable from device code.

---

## 4.8 Practical Rules

```text
- Default to DeviceBuffer<T> / HostBuffer<T> and parallel_for<>; reach for
  thrust:: / tbb:: only when an abstraction genuinely does not exist.
- Keep backend-specific code behind ATLAS_TASKING_CUDA / ATLAS_TASKING_TBB.
- Give every device-reachable function the right host/device attributes.
- Engine translation units are .cu and live in src/atlas/, mirroring the
  include/atlas/ directory layout.
- The scalar type is float; do not add new template <typename T> scalar
  parameterization.
```

See [06-conventions.md](06-conventions.md) for the file and module conventions
that complement these portability rules.
