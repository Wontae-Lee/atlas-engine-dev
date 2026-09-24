# Sink

The sink module decides whether a particle should be **removed** (despawned)
because it reached or crossed a boundary `Unit`. Like the collider it is a
tagged-union of self-contained leaves.

## Files

| File | Role |
|---|---|
| `include/atlas/sink/sink.h` | `Sink` umbrella (`DeviceVariant`) + `ConceptSink` |
| `include/atlas/sink/sink_type.h` | `enum class SinkType { surface, volume, tracing }` |
| `include/atlas/sink/surface_sink.h` | `SurfaceSink` leaf + `Builder` |
| `include/atlas/sink/volume_sink.h` | `VolumeSink` leaf + `Builder` |
| `include/atlas/sink/tracing_sink.h` | `TracingSink` leaf + `Builder` |
| `src/atlas/sink/*.cu` | leaf `Builder` implementations |

## Design

`Sink` is a **`DeviceVariant`**; the leaves are trivially copyable, so
`DeviceBuffer<Sink>` works and dispatch runs on the device.

```cpp
class Sink {
    SinkType type;
    union { SurfaceSink surface; VolumeSink volume; TracingSink tracing; };
    bool despawn(position, velocity, dt) const;   // visit → leaf, fallback false
    void advance(dt);                             // apply → leaf (mutable)
};
```

### Leaf contract — `ConceptSink`

```cpp
template <typename S>
concept ConceptSink = requires(S sink, const Float3 vec, float dt) {
    { sink.despawn(vec, vec, dt) } -> std::same_as<bool>;
    { sink.advance(dt) }           -> std::same_as<void>;
};
```

Each leaf owns a `Unit` (and, for surface/volume, a `tolerance`) and tests the
particle against **its own** unit geometry, transforming the particle into the
unit's local frame via `_unit.sync().sync_to_local(...)`:

| Leaf | `despawn` test |
|---|---|
| `SurfaceSink` | `geometry().is_on_surface(local, tolerance)` |
| `VolumeSink`  | `geometry().is_inside(local, tolerance)` |
| `TracingSink` | `_unit.trace(Ray(position, velocity))` hit within `speed·dt` |

`advance(dt)` advances the owned unit. As with the collider, the per-particle
`parallel_for` and the removal/compaction of despawned particles happen
**outside** the sink (compaction is a caller concern).

Each leaf builder requires a `Unit`. Surface and volume builders additionally
validate finite, non-negative `tolerance`; tracing has no tolerance setting.
`build()` moves the Unit into the sink and clears the builder's staged Unit.
The resulting `Sink` owns that Unit by value and can be copied into a backend
buffer; its geometry queries use the current pose.

## Adding a leaf

Same recipe as the collider: a trivially-copyable leaf satisfying `ConceptSink`,
a `SinkType` value, a union member + `DeviceVariantCase`, and a
`static_assert(ConceptSink<NewLeaf>)`.
