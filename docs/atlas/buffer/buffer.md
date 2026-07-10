# Buffer

The buffer module is the engine's two owning-container aliases: `HostBuffer<T>`
for host-resident storage and `DeviceBuffer<T>` for the parallel backend's
storage. It has no runtime behavior and no place in the step pipeline — it is the
foundation the rest of the engine names instead of writing `thrust::device_vector`
or `std::vector` directly. Together with `memory/`, `parallel/`, and `scan/` it is
one of the few corners that know which backend is compiled; every other module
uses these two names and stays backend-agnostic.

## Files

| File | Role |
|---|---|
| `include/atlas/buffer/device_buffer.h` | `DeviceBuffer<T>` alias — the backend's device-side owning container |
| `include/atlas/buffer/host_buffer.h` | `HostBuffer<T>` alias — host-side owning container that copies cheaply to/from a `DeviceBuffer` |

There is no `src/atlas/buffer/`; both files are header-only alias definitions.

## The two aliases

Each alias resolves to a different concrete type per backend, selected by the
`ATLAS_BACKEND_CUDA` macro that CMake defines (exactly one of
`ATLAS_BACKEND_CUDA` / `ATLAS_BACKEND_TBB` is present):

```cpp
#if defined(ATLAS_BACKEND_CUDA)
template <typename T> using DeviceBuffer = thrust::device_vector<T>;
template <typename T> using HostBuffer   = thrust::host_vector<T>;
#else
template <typename T> using DeviceBuffer = std::vector<T>;
template <typename T> using HostBuffer   = std::vector<T>;
#endif
```

| | CUDA backend | Host backend |
|---|---|---|
| `DeviceBuffer<T>` | `thrust::device_vector<T>` (GPU memory) | `std::vector<T>` |
| `HostBuffer<T>` | `thrust::host_vector<T>` | `std::vector<T>` |

### `DeviceBuffer<T>` — the thing a kernel may not capture

Under CUDA, `DeviceBuffer<T>` is `thrust::device_vector<T>`: the engine's handle
to GPU-resident storage. It owns its allocation, frees it on destruction, and
resizes with device-side reallocation. A device kernel cannot dereference the
container itself; device-reachable code reaches its elements through
`raw_pointer_cast` on `begin()`/`data()`, or through the transfer helpers in
`memory/copy.h`. Under the host backend there is no separate device memory, so it
degrades to `std::vector<T>` and every "device" operation runs on CPU threads —
but the same discipline survives: a `DeviceBuffer` is still the thing a kernel may
not capture, and callers still reach elements through `raw_pointer_cast` rather
than through the container.

The load-bearing property for the whole architecture is that under CUDA
`thrust::device_vector`'s **copy constructor and copy assignment are host-only**.
That is exactly why leaf types that own a `DeviceBuffer` are held move-only
through `HostVariant` (`include/atlas/core/host_variant.h`) rather than
`DeviceVariant` — the copy that `DeviceVariant`'s device-capturable value would
require does not exist on the device.

Element type `T` must satisfy thrust's requirements for device storage; for a
buffer that is memcpy'd to or from the host it must be trivially copyable.

### `HostBuffer<T>` — why it stays `thrust::host_vector` under CUDA

The subtle one. It would be natural to make `HostBuffer<T>` simply
`std::vector<T>` in both backends, since it lives in host memory either way. It is
deliberately **not**. Under CUDA it is `thrust::host_vector<T>`, and the reason is
the one operation callers rely on: constructing a `HostBuffer` from a
`DeviceBuffer`'s iterator pair is a **single bulk copy**, never a per-element
transfer.

```cpp
const HostBuffer<int> flags(fluid.active().begin(), fluid.active().end());
```

Built this way, `thrust::host_vector` recognizes the device iterators and issues
one device-to-host `cudaMemcpy`. A `std::vector` constructed from the same
`thrust::device_vector` iterators would instead dereference each element on its
own — and under CUDA every such dereference is its own `cudaMemcpy`. That pattern
is exactly how the observer and the serializer pull GPU state back to host memory
for CSV rows and protobuf snapshots (`src/atlas/observer/observer.cu`,
`src/atlas/serialization/protobuf_snapshot.cpp` — roughly two dozen call sites
build a `HostBuffer` straight from a `DeviceBuffer`'s `begin()/end()`).

Because `HostBuffer` is a value container, its elements must be copyable. That is
precisely why the move-only leaves — the ones that own a `DeviceBuffer` — cannot
live in a `HostBuffer` of leaves and are instead held as host smart pointers
(the `…HostPtr` = `host_shared_ptr` / `host_unique_ptr` aliases, e.g.
`SolverHostPtr`, `SourceHostPtr`, `GeneratorHostPtr`). `HostBuffer<T>` is used
directly only for copyable payloads: the trivially-copyable leaves that *do* live
in a `DeviceVariant` (`HostBuffer<Collider>`, `HostBuffer<Sink>` in
`System::Builder`), scalar staging arrays (`HostBuffer<float>` species ratios in
the generators), mesh triangles (`HostBuffer<TriangleContainer4>`), and the
host-side landing buffers of the observer and serializer.

## Host vs. device

Both aliases are pure host-side type declarations — they define no functions and
run no code themselves. What runs is whatever the underlying container does:
`DeviceBuffer` allocations and resizes touch device memory under CUDA; the
transfers between the two happen only through the explicit `memory/copy.h`
helpers or through the bulk-copy constructor described above. Nothing here is
device-callable.

## Deliberately absent

- **No wrapper type, no methods of its own.** The module is two `using` aliases,
  not a `Buffer<T>` class. This is a choice: the engine wants the full,
  familiar `thrust`/`std::vector` interface at every call site (`begin()`,
  `size()`, `resize()`, iterator-pair construction) rather than a thin wrapper
  that would have to re-expose it. The backend-specific behavior that *does* need
  a named home — transfers, raw-pointer access, parallel algorithms, scans —
  lives in the sibling `memory/`, `parallel/`, and `scan/` headers, not here.
- **No `src/atlas/buffer/`.** Header-only by nature; there is nothing to define.
- **No third "unified/managed memory" alias.** The two-container model (explicit
  host buffer, explicit device buffer, explicit copies between them) is the whole
  design; CUDA unified memory is not used.

## Not implemented

Nothing. The buffer module declares no type, function, parameter, or case that
the engine fails to produce, consume, or reach. Both aliases are load-bearing and
widely used: `include/atlas/atlas.h` pulls in both headers, `DeviceBuffer<T>`
appears in ~39 files under `include/`+`src/`, and `HostBuffer<T>` in ~26 files
across `include/`, `src/`, `tests/`, and `benchmarks/`. There is no
declared-but-unproduced, accepted-but-ignored, uncalled, or stubbed surface in
this module.

## Extending

There is almost nothing to extend — the module is intentionally two lines of
alias per backend. The only kind of change that belongs here is adding a new
backend or a new owning-container family, and it must preserve the invariants the
rest of the engine assumes:

- Any new `DeviceBuffer` backing must keep its copy operations host-only (or
  otherwise non-device-capturable) so the `HostVariant`-vs-`DeviceVariant`
  decision stays valid; and its elements must be reachable through
  `raw_pointer_cast` for device code.
- Any new `HostBuffer` backing must make **iterator-pair construction from a
  `DeviceBuffer` a single bulk transfer**, not a per-element copy — this is the
  reason the alias tracks the backend at all.

A new backend is introduced by adding a branch under a new `ATLAS_BACKEND_*`
macro in *both* headers (and the matching branches in `memory/`, `parallel/`,
`scan/`), keeping exactly one backend defined per build. No `static_assert` or
`validate()` guards this module; correctness is enforced downstream — a
`DeviceBuffer`-owning leaf placed in a `DeviceVariant` fails to compile because
the required copy is host-only, and that compile error is the check.
