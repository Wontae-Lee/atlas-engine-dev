# Source

The source module **emits** new particles from a boundary `Unit`: it caches the
accepted surface/interior sample points of the unit and writes them, transformed
to world space, into a fluid position buffer.

## Files

| File | Role |
|---|---|
| `include/atlas/source/source.h` | `Source` umbrella (`HostVariant`) + `ConceptSource` |
| `include/atlas/source/source_type.h` | `enum class SourceType { surface, volume }` |
| `include/atlas/source/surface_source.h` | `SurfaceSource` leaf + `Builder` |
| `include/atlas/source/volume_source.h` | `VolumeSource` leaf + `Builder` |
| `src/atlas/source/*.cu` | leaf `Builder` + cache build + spawn kernel |

## Why `HostVariant`, not `DeviceVariant`

Each source leaf **owns a `DeviceBuffer<Float3>` cache** of accepted local sample
points (`std::vector` on TBB, `thrust::device_vector` on CUDA). Its copy/move/destructor are
**host-only**, and `DeviceVariant`'s union machinery is `__host__ __device__`. If
a `DeviceVariant` wrapped such a leaf, nvcc would instantiate the union
construct/copy/destroy for the **device** and try to call `device_vector`'s
host-only members there — emitting `#20014 "calling a __host__ function from a
__host__ __device__ function is not allowed"`.

`Source` therefore uses **`HostVariant`** (`core/host_variant.h`): the same tagged
union, but with `ATLAS_HOST`-only members, so no device instantiation happens.
It is host-side and move-only. (Verified: `DeviceVariant`+`device_vector` emits
`#20014`; `HostVariant` emits none.)

```cpp
class Source {              // host-only, move-only
    SourceType type;
    union { SurfaceSource surface; VolumeSource volume; };
    int  spawn(FluidPositionState* positions, std::size_t offset) const;  // visit → leaf
    void advance(float dt);                                               // apply → leaf
};
```

The special members (default/payload/move ctors, move-assign, destructor) and
dispatch are all delegated to `SourceVariant` (a `HostVariant`).

### Leaf contract — `ConceptSource`

```cpp
template <typename S>
concept ConceptSource = requires(S s, const S cs, FluidPositionState* p, std::size_t off, float dt) {
    { cs.spawn(p, off) } -> std::same_as<int>;
    { s.advance(dt) }    -> std::same_as<void>;
};
```

Each leaf owns a `Unit`, a `tolerance`, a `spacing`, and the
`DeviceBuffer<Float3>` cache. The cache is built **once** at `build()` time:
grid-sample the geometry bound (`sample_axis_count(lower, upper, spacing)`) and
keep the points passing the leaf's predicate:

| Leaf | accept predicate |
|---|---|
| `SurfaceSource` | `geometry().is_on_surface(local, tolerance)` |
| `VolumeSource`  | `geometry().is_inside(local, tolerance)` |

`spawn(positions, offset)` launches a **device** kernel: for each cached local
point it writes `unit.sync().sync_to_world(point)` into `positions->data()` at
`offset + i` and returns the number written. `advance(dt)` advances the unit; the
cache is in local space so it stays valid.

Both leaf builders take a `Unit`, `tolerance`, and `spacing`.
`Builder::validate()` requires a Unit, finite non-negative tolerance, and finite
positive spacing. `build()` moves the Unit into the source, creates its sample
cache, and resets the builder. The resulting `Source` owns its cache; the
spawn kernel borrows its buffer during the call.

## Emission flow

`System::emit()` loops over sources on the host; each `source.spawn(...)` and
each generator (`generator.generate(...)`) launches its own device kernel. The
host-side `HostVariant` dispatch is one `switch`/`visit` per source per step —
negligible; all heavy work stays on the device.
