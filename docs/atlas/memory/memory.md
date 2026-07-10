# Memory

The memory module is the engine's **pointer and transfer vocabulary**. It sits
underneath every module rather than at a step in the pipeline: it supplies the
smart-pointer aliases the builders return (`host_shared_ptr`, `host_unique_ptr`,
`device_shared_ptr`), the `raw_pointer_cast` that turns a backend fancy pointer
into a bare address a kernel can capture, and the `copy_*` helpers that move a
handful of elements across the host/device boundary. Everything here is
header-only; there is no `src/atlas/memory/`.

Like `buffer/`, `parallel/`, and `scan/`, these are backend-split headers: each
one has a `#ifdef ATLAS_BACKEND_CUDA` shape (Thrust + CUDA managed memory) and a
plain-host shape (`std::` containers, raw pointers), and CMake defines exactly
one backend.

## Files

| File | Role |
|---|---|
| `include/atlas/memory/memory.h` | Smart-pointer aliases + `make_*` factories; the managed `device_shared_ptr` / `device_refcount` under CUDA |
| `include/atlas/memory/copy.h` | `copy_device_to_host` / `copy_host_to_device` overloads over raw pointers, `device_ptr`, and `DeviceBuffer` |
| `include/atlas/memory/raw_pointer_cast.h` | `device_ptr<T>` alias + `raw_pointer_cast` to strip the fancy-pointer wrapper |

## `raw_pointer_cast.h` — the fancy-pointer boundary

`device_ptr<T>` is `thrust::device_ptr<T>` under CUDA and a plain `T*` on the
host backend. `DeviceBuffer<T>::data()` hands out a `device_ptr<T>`; a device
lambda cannot capture a `thrust::device_ptr` and dereference it as a raw
address, so kernel code first calls `raw_pointer_cast` to get the bare `T*`.

```cpp
template <typename T> constexpr T*       raw_pointer_cast(T* p) noexcept;
template <typename T> constexpr const T* raw_pointer_cast(const T* p) noexcept;
#if defined(ATLAS_BACKEND_CUDA)
template <typename T> T*       raw_pointer_cast(device_ptr<T> p) noexcept;
template <typename T> const T* raw_pointer_cast(device_ptr<const T> p) noexcept;
#endif
```

The raw-pointer overloads are the identity; they exist so a call site can write
`raw_pointer_cast(buffer.data())` without knowing which backend produced the
pointer. This is the most-used symbol in the module (~110 call sites across the
generators, solvers, views, and codecs). All overloads are
`ATLAS_ALL_DEVICE` — callable from host and device.

## `copy.h` — small synchronous transfers

Every overload is `ATLAS_HOST ATLAS_FORCE_INLINE` and issues **one** synchronous
`cudaMemcpy` (via `thrust::copy_n`) under CUDA, or a `std::copy_n` on the host
backend. `count == 0` short-circuits before touching `src`. These are for
pulling a scalar or a small run back to the host to drive control flow, not for
bulk data movement.

```cpp
template <typename T> void copy_device_to_host(const T* src, T* dst, std::size_t count);
template <typename T> void copy_device_to_host(device_ptr<const T> src, T* dst, std::size_t count);   // CUDA only
template <typename T> void copy_device_to_host(const DeviceBuffer<T>& src, T* dst, std::size_t count);

template <typename T> void copy_host_to_device(const T* src, T* dst, std::size_t count);
template <typename T> void copy_host_to_device(const T* src, device_ptr<T> dst, std::size_t count);   // CUDA only
template <typename T> void copy_host_to_device(const T* src, DeviceBuffer<T>& dst, std::size_t count);
```

The two live call sites both use the raw-pointer `copy_device_to_host` to read a
single reduced `int` back to the host:

- `src/atlas/fluid/fluid.cu:95` — the survivor count that drives compaction.
- `src/atlas/solver/dsmc/dsmc_solver.cu:199` — the candidate-pair count that
  decides whether the collision kernel launches.

In both, the raw device address comes from
`raw_pointer_cast(buffer.data())`, so the raw-pointer overload is selected and
the `device_ptr` / `DeviceBuffer` overloads are never reached (see *Not
implemented*).

## `memory.h` — pointer aliases and the managed pointer

Host aliases and factories are trivial pass-throughs and are used everywhere the
builders live (`make_host_shared` ~100 sites, `make_host_unique` ~13):

```cpp
template <typename T> using host_shared_ptr = std::shared_ptr<T>;
template <typename T> using host_unique_ptr = std::unique_ptr<T>;
template <typename T, typename... Args> host_shared_ptr<...> make_host_shared(Args&&...);
template <typename T, typename... Args> host_unique_ptr<...> make_host_unique(Args&&...);
```

The substance of the header is `device_shared_ptr<T>`, a hand-rolled
reference-counted owning pointer whose payload **and** strong counter both live
in `cudaMallocManaged` storage. That is the whole reason it exists rather than a
`std::shared_ptr`: a copy of the pointer object can be captured **by value** into
a device lambda and dereferenced on the GPU, while the refcount stays coherent
across host and device because the bump/drop go through
`ATLAS_ATOMIC_ADD`/`ATLAS_ATOMIC_SUB` (true `atomicAdd`/`atomicSub` in device
code, plain arithmetic on the host).

`device_refcount` is a non-owning handle over the managed `int*` counter; it does
the atomic arithmetic but never allocates or frees. A null counter makes every
operation a safe no-op.

Teardown is deliberately **asymmetric**:

- On the **host**, dropping the last reference runs `~T()` and `cudaFree` on both
  the object and the counter (`release_host_side`).
- On the **device**, destructors and assignments only decrement the counter —
  `cudaFree` is illegal from a kernel, so managed storage is reclaimed by the
  host copy that outlives the device work, never by a device-side drop to zero.

`make_device_shared` allocates both blocks with `cudaMallocManaged`, placement-
constructs the object, seeds the count to 1, and rolls back a half-succeeded
allocation, returning an **empty** pointer on failure (the caller must check).

On the host backend, `device_shared_ptr` collapses to `std::shared_ptr` and
`make_device_shared` to `std::make_shared`, so call sites compile unchanged
without any CUDA toolkit.

## Deliberately absent

- **No thread-safe host teardown.** `release_host_side` reads the counter,
  branches, then frees without a lock; the header states device execution is
  expected to have been synchronized before the owning host pointer is
  destroyed. This is a documented assumption, not an oversight.
- **No weak pointer, no aliasing constructor, no array form.** `device_shared_ptr`
  is the minimal shape needed to carry a device-capturable owner; the standard
  `shared_ptr` surface beyond strong ownership is intentionally not reproduced.
- **`copy_*` are single synchronous transfers, not a stream/async API.** They
  exist to pull a scalar back for host control flow; bulk movement goes through
  `DeviceBuffer` directly.

## Not implemented

The managed `device_shared_ptr` machinery is fully written and documented but
has **no producer anywhere in the engine**, and several `copy.h` overloads and
one `device_refcount` method have no caller. Nothing here is broken; it is a
provided facility waiting for a user, or a symmetry-for-completeness overload.

| What | Where | Evidence | Verdict |
|---|---|---|---|
| `make_device_shared` | `memory.h:384` | Zero calls in `include`, `src`, `tests`, `benchmarks`. Nothing ever constructs a managed `device_shared_ptr`. | Declared, never called — the only factory that would produce the managed pointer |
| `device_shared_ptr` (managed CUDA form) | `memory.h:154` | Named only in ~30 `using <Type>DevicePtr = device_shared_ptr<...>` aliases; grep for `DevicePtr` outside those `using` lines returns nothing — no variable, member, return type, or argument of any alias exists. With no `make_device_shared` caller, no instance is ever produced. | No producer — a device-capturable owner facility that no module has adopted; every current owner is a `HostVariant`/`host_shared_ptr` instead |
| Every `*DevicePtr` alias (e.g. `ColliderDevicePtr`, `SolverDevicePtr`, `SourceDevicePtr`, …) | e.g. `collider/collider.h:290`, `solver/solver.h:93`, `source/source.h:256` | Defined once each, referenced nowhere. | Dead alias — the intended extension point for the unused `device_shared_ptr` |
| `device_refcount::release()` | `memory.h:106` | Its own `@warning` says so: `device_shared_ptr` inlines its decrement in `release_host_side` and the destructor instead. No external caller. | Declared, never called — superseded by inlined logic |
| `copy_host_to_device` (all three overloads) | `copy.h:100,124,144` | Zero callers anywhere. The engine uploads through `DeviceBuffer` construction/resize, not this helper. | Declared, never called — provided for symmetry with `copy_device_to_host` |
| `copy_device_to_host(device_ptr<const T>, …)` and `copy_device_to_host(const DeviceBuffer<T>&, …)` | `copy.h:59,79` | The two live callers both pass a raw `int*` from `raw_pointer_cast(...)`, selecting the raw-pointer overload; these two are never reached. | Declared, never called — overload-set completeness |

## Extending

- **To make a module carry a device-capturable owner**, wrap it in
  `device_shared_ptr<T>` via `make_device_shared<T>(args...)`. `T` must be
  destructible (`static_assert` in `make_device_shared`) and, to actually be
  dereferenced inside a kernel, trivially usable from device code. This is the
  path the unused `*DevicePtr` aliases were written for; adopting one is what
  would turn the *Not implemented* rows above into live code.
- **To add a transfer shape**, add an overload to `copy.h` alongside the
  existing three per direction, guarding any `device_ptr`-typed overload with
  `#if defined(ATLAS_BACKEND_CUDA)` (on the host backend `device_ptr<T>` *is*
  `T*`, so an unguarded overload would redeclare the raw-pointer one).
- **To support a new fancy pointer in `raw_pointer_cast`**, add the overload
  under the CUDA guard for the same reason; keep the raw-`T*` identity overloads
  unguarded so host-backend call sites still compile.
- There is no `validate()` or umbrella here — this module is plumbing, not a
  tagged-union leaf family, so the mistake caught at compile time is the
  duplicate-overload / redeclaration one the backend guards prevent, plus
  `make_device_shared`'s destructibility `static_assert`.
</content>
</invoke>
