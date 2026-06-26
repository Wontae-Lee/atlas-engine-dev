# 4. Backend Model and Portability

Atlas compiles from one source tree for either the TBB (CPU) or CUDA (GPU)
backend. This document describes how the backend is selected and how code stays
backend-agnostic.

---

## 4.1 Exactly One Backend

Exactly one backend must be enabled at configure time:

```text
ATLAS_USE_TBB = ON,  ATLAS_USE_CUDA = OFF      (CPU / TBB)
ATLAS_USE_TBB = OFF, ATLAS_USE_CUDA = ON       (GPU / CUDA)
```

Enabling both, or neither, is a configuration error. Compile-time backend
branches key off these macros:

```text
ATLAS_TASKING_TBB     active when the TBB backend is selected
ATLAS_TASKING_CUDA    active when the CUDA backend is selected
```

---

## 4.2 Shared Aliases

Backend differences in storage and ownership are hidden behind shared aliases.
Prefer these over backend-specific types:

| Alias | TBB backend | CUDA backend |
|---|---|---|
| `DeviceBuffer<T>` | `std::vector<T>` | `thrust::device_vector<T>` |
| `HostBuffer<T>` | `std::vector<T>` | `thrust::host_vector<T>` |
| `device_shared_ptr<T>` | `std::shared_ptr<T>` | CUDA-aware managed ownership |
| `host_shared_ptr<T>` | `std::shared_ptr<T>` | `std::shared_ptr<T>` |

`DeviceBuffer<T>` holds data the active backend processes (on the GPU under
CUDA, on the CPU under TBB). `HostBuffer<T>` is host-resident staging/config
data. `device_shared_ptr<T>` provides shared ownership that remains valid
across kernel boundaries on CUDA.

---

## 4.3 Parallel Dispatch

Backend-agnostic parallelism goes through the `parallel_for` family and the
companion algorithms, parameterized by an execution policy:

```cpp
parallel_for<ExecutionPolicy::device>(begin, end, func);
```

`ExecutionPolicy` selects `serial`, `host`, or `device` execution; the
implementation dispatches at compile time to the active backend (Thrust on
CUDA, TBB on the CPU). Companion primitives follow the same policy-parameterized
shape:

```text
parallel_for, parallel_fill, parallel_sort   (parallel/)
exclusive_scan                               (scan/)
transform, transform_reduce                  (transform/)
remove_if                                     (remove/)
```

Use these instead of writing raw `thrust::` or `tbb::` calls in core code.

---

## 4.4 Portability Macros

`include/atlas/core/macros.h` provides the host/device portability layer. Use
these macros when appropriate rather than backend-specific attributes:

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
host/device attributes; a header-only hot-path function typically combines an
attribute with `ATLAS_FORCE_INLINE`.

---

## 4.5 Device-Callable Operators

Virtual interfaces (for example `Geometry<T>`, `Generator<T>`, `Solver<T>`,
`Searcher<T>`) are convenient on the host but cannot be called through vtables
inside kernels. The pattern Atlas uses is a **device-callable operator**: a
value-type union/variant that encodes the concrete behavior without virtual
dispatch.

```text
Geometry<T>      ->  GeometryOperator<T>     (geometry/)
Generator<T>     ->  GenerateOperator<T>     (generator/)
surface model    ->  SurfaceInteractionKernel<T>  (collider/interaction/)
```

Host code builds the operator from a polymorphic description; kernels receive
the operator by value and dispatch on its tag. When adding a new geometry,
generator, or interaction model, extend the matching operator union so the new
case is reachable from device code.

---

## 4.6 Practical Rules

```text
- Default to DeviceBuffer<T> / HostBuffer<T> and parallel_for<>; reach for
  thrust:: / tbb:: only when an abstraction genuinely does not exist.
- Keep backend-specific code behind ATLAS_TASKING_CUDA / ATLAS_TASKING_TBB.
- Give every device-reachable function the right host/device attributes; do
  not leave declarations and definitions inconsistent across backends.
- CUDA translation units use .cu; CPU translation units use .cpp. Header-only
  template code stays in .h/.hpp pairs and compiles under both backends.
```

See [06-conventions.md](06-conventions.md) for the file and module conventions
that complement these portability rules.
