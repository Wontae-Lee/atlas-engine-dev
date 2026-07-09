# Codec

A codec decides, per cell, **which solver runs there**. It reads the universe's
per-cell states and writes the decision into `UniverseAllocatedSolverState`,
which the orchestrator then dispatches on.

`Codec` follows the same tagged-union leaf pattern as `Source` and `Generator`:
because its leaf owns `DeviceBuffer` tables, the umbrella is a **`HostVariant`**
(host-only, move-based), not a `DeviceVariant`.

## Files

| File | Role |
|---|---|
| `include/atlas/codec/codec.h` | `ConceptCodec` + `Codec` umbrella (`HostVariant`) |
| `include/atlas/codec/codec_type.h` | `enum class CodecType { knudsen }` |
| `include/atlas/codec/knudsen_codec.h` | `KnudsenCodec` leaf + `Builder` |
| `src/atlas/codec/knudsen_codec.cu` | leaf + `Builder` implementation |

## `allocate`

There is **one** operation, not a separate encode/decode pair — computing a
cell's Knudsen number and bucketing it into a solver index happen in a single
kernel, so the intermediate never needs to be stored.

```cpp
void allocate(const UniverseTemperatureState* temperature,
              const UniverseNumberParticleState* number_particle,
              UniverseAllocatedSolverState* allocated_solver) const;
```

The universe states are passed in rather than reached through a `Universe`
handle, so a codec leaf is self-contained and needs no `Fluid`, `Searcher`, or
probe struct. A null argument, an empty split table, or a size mismatch between
`number_particle` and `allocated_solver` makes the call a no-op.

`temperature` is accepted but unused by `KnudsenCodec` — it is part of the
operation's shape for leaves that need it.

## `KnudsenCodec`

Every cell is processed; none is skipped. Two `__host__ __device__` member
functions carry the physics, both reading the leaf's own state:

```cpp
float knudsen_number(float particle_count) const;   // uses the four representative scalars
int   solver_index(float kn) const;                 // buckets against _kn_split
```

`knudsen_number` turns the cell's particle count into a number density
(`particle_count * representative_statistical_weight / representative_cell_volume`),
derives the mean free path from the collision cross-section, and divides by the
characteristic length. `solver_index` then buckets that number against the split
table, yielding one of five solver indices.

The split table is **fixed** at `{0.01, 0.1, 1.0, 10.0}` — `split_count` is a
`static constexpr int` of 4 and there is no way to override it. It lives in a
`Container<float, split_count>`, not a `DeviceBuffer<float>`, because the leaf
must be **trivially copyable** for the device lambda to capture it, exactly as
the collider leaves are.

`allocate()` copies the leaf into the kernel and calls both:

```cpp
const KnudsenCodec codec = *this;
parallel_for<ExecutionPolicy::device>(0, cell_count, [=] ATLAS_ALL_DEVICE(const int cell) {
    allocated_solver_ptr[cell] = codec.solver_index(codec.knudsen_number(number_particle_ptr[cell]));
});
```

```cpp
auto codec = KnudsenCodec::builder()
    .with_representative_characteristic_length(1.0f)
    .with_representative_collision_cross_sectional_area(1.0f)
    .with_representative_statistical_weight(fluid.statistical_weight())
    .with_representative_cell_volume(universe.cell_volume())
    .build();
```

All four scalars are **representative values** standing in for the whole domain,
so the leaf holds them by value instead of chasing a handle each step.

## `Codec`

```cpp
template <typename C>
concept ConceptCodec = requires(const C codec,
                                const UniverseTemperatureState* temperature,
                                const UniverseNumberParticleState* number_particle,
                                UniverseAllocatedSolverState* allocated_solver) {
    { codec.allocate(temperature, number_particle, allocated_solver) } -> std::same_as<void>;
};

class Codec {
public:
    CodecType type;
    union { KnudsenCodec knudsen; };
    void allocate(...) const;              // dispatch via HostVariant::apply
};
```

`Codec` is **move-only** (copy is deleted) with move/destroy delegated to
`HostVariant`. Construct from a leaf: `Codec c(KnudsenCodec::builder()…build());`

## Adding a leaf

Add a `CodecType` value, a leaf header (implement `allocate`, add a `Builder`), a
union member and a `HostVariantCase`. The `static_assert(ConceptCodec<…>)` in
`codec.h` catches a missing or mistyped `allocate`.
