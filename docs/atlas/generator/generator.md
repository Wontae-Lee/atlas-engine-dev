# Generator

The generator module **fills the initial velocity and species** of freshly
spawned particles. It is the emit-phase partner of the
[source](../source/source.md): the source writes positions into a contiguous
range of the fluid buffers, then the paired generator writes the velocity and
species id for exactly those slots. Like the other emit/solve modules it follows
the **tagged-union leaf** pattern — one concrete umbrella type wraps one of
several self-contained leaf types and dispatches to it.

## Files

| File | Role |
|---|---|
| `include/atlas/generator/generator.h` | `Generator` umbrella (`HostVariant`) + `ConceptGenerator` |
| `include/atlas/generator/generator_type.h` | `enum class GeneratorType { uniform, jittering, maxwell_sigma, maxwell_boltzmann }` |
| `include/atlas/generator/uniform_generator.h` | `UniformGenerator` leaf + `Builder` |
| `include/atlas/generator/jittering_generator.h` | `JitteringGenerator` leaf + `Builder` |
| `include/atlas/generator/maxwell_sigma_generator.h` | `MaxwellSigmaGenerator` leaf + `Builder` |
| `include/atlas/generator/maxwell_boltzmann_generator.h` | `MaxwellBoltzmannGenerator` leaf + `Builder` |
| `src/atlas/generator/*.cu` | leaf `Builder` + validation + the `generate` device kernel |

## Why `HostVariant`, not `DeviceVariant`

Every generator leaf **owns one or more `DeviceBuffer<float>` members** — the
species selection weights (`_species_ratios`), the parallel species-id table
(`_species_numbers`), and, for `MaxwellBoltzmannGenerator`, the per-species
masses (`_species_mass`). A `DeviceBuffer` is a `thrust::device_vector`, whose
copy/move/destructor are **host-only**, so the leaf is move-only and cannot live
in a device-side union.

`Generator` therefore uses **`HostVariant`** (`core/host_variant.h`): the same
tagged union as `DeviceVariant`, but with `ATLAS_HOST`-only special members, so
nvcc never instantiates the union's construct/copy/destroy for the device and
never tries to call `device_vector`'s host-only members there. This is the same
reason [`Source`](../source/source.md) is a `HostVariant` (and the contrast with
the trivially-copyable [`Collider`](../collider/collider.md) /
[`Sink`](../sink/sink.md), which are `DeviceVariant`). The dispatch machinery
runs on the host; the buffers live on the device; the per-particle sampling
happens inside each leaf's device kernel.

```cpp
class Generator final {                 // host-only, move-only
    GeneratorType type = GeneratorType::uniform;
    union {
        UniformGenerator          uniform;
        JitteringGenerator        jittering;
        MaxwellSigmaGenerator     maxwell_sigma;
        MaxwellBoltzmannGenerator maxwell_boltzmann;
    };
    int  generate(FluidVelocityState*, FluidSpeciesState*,
                  std::size_t offset, std::size_t count) const;   // visit → leaf
    void set_bulk_velocity(const Float3& bulk_velocity) noexcept;  // apply → leaf
};
```

The copy constructor and copy-assignment are **deleted**; the default/payload/
move constructors, move-assign, and destructor are all thin forwarders into
`GeneratorVariant` (a `HostVariant`), so the correct union member's special
functions run for whichever tag is live. `GeneratorGenerate` and
`GeneratorSetBulkVelocity` are the visitor function objects the umbrella hands to
`GeneratorVariant::visit` / `::apply`. `generate` returns `0` (the visit
fallback) if `type` matches no case. The default constructor installs a default
`UniformGenerator`.

### Leaf contract — `ConceptGenerator`

Every leaf must satisfy `ConceptGenerator`, enforced by `static_assert` for all
four leaves in `generator.h`:

```cpp
template <typename G>
concept ConceptGenerator = requires(G generator,
                                    const G const_generator,
                                    FluidVelocityState* velocities,
                                    FluidSpeciesState* species,
                                    const Float3 bulk_velocity,
                                    std::size_t offset,
                                    std::size_t count) {
    { const_generator.generate(velocities, species, offset, count) } -> std::same_as<int>;
    { generator.set_bulk_velocity(bulk_velocity) }                   -> std::same_as<void>;
};
```

- **`generate(velocities, species, offset, count) → int`** (const) — writes
  `[offset, offset + count)` of the velocity and species buffers and returns how
  many slots it actually filled. Each leaf independently:
  - returns `0` on a null `velocities`/`species` pointer or a zero `count`;
  - computes `capacity = min(velocity_buffer.size(), species_buffer.size())` —
    every written index must be valid in **both** buffers, so the shorter one
    bounds the write;
  - returns `0` if `offset >= capacity`;
  - clamps the request to `writable = min(count, capacity - offset)` and writes
    that many slots, returning `writable` (which may be less than `count`).
- **`set_bulk_velocity(bulk_velocity)`** (mutable) — retargets the constant drift
  added to every sampled velocity. Each leaf stores it in `_bulk_velocity`; the
  umbrella exposes the same through `GeneratorSetBulkVelocity`.

The per-particle loop lives **inside** each leaf: `generate` launches an
`atlas::parallel_for<ExecutionPolicy::device>` over `writable` slots with a
capture-by-value device lambda, so all owned scalars/pointers are copied into the
device closure.

## The leaves

All four leaves share the same shape: two (or three) device buffers, a stored
`_temperature` (kelvin, default `273.15f`), a `Float3 _bulk_velocity`, and an
`unsigned int _seed`. They differ only in the **velocity distribution** they draw
and how they resolve species. In every leaf the sampled velocity has the bulk
drift added on top.

### `UniformGenerator`

Each velocity component is drawn from a uniform distribution over
`[min_value, max_value]`, then shifted by the drift:

```cpp
velocity = sample_uniform_vector(engine, min_value, max_value) + bulk_velocity;
```

Non-physical and the simplest model. `_temperature` is carried and validated but
**does not** influence the sampled velocity. Parameters: `min_value`,
`max_value` (with `min_value <= max_value` enforced at build). Owns two device
buffers.

### `JitteringGenerator`

A constant base value per component, perturbed by a small symmetric uniform
jitter, plus drift:

```cpp
velocity = Float3(base_value, base_value, base_value)
         + sample_uniform_vector(engine, -radius, radius)   // radius = std::abs(_jitter_radius)
         + bulk_velocity;
```

The kernel takes `|jitter_radius|`, so a negative setting is treated
symmetrically. This is a "roughly here, with a little spread" model;
`_temperature` is stored/validated but unused. Parameters: `base_value`,
`jitter_radius`. Owns two device buffers.

### `MaxwellBoltzmannGenerator`

The only leaf that turns `_temperature` into physics. For the chosen species of
molecular mass `m`, the per-component standard deviation is derived from the
temperature and mass exactly as:

```cpp
sigma    = sqrt_nonnegative(boltzmann_constant * temperature / molecular_mass);   // sqrt(k_B * T / m)
velocity = sample_normal_vector(engine, sigma) + bulk_velocity;
```

When either `temperature` or `molecular_mass` is non-positive (sigma undefined),
the particle receives **only** the bulk drift. `sample_normal_vector` produces an
isotropic Gaussian (see [sampling](../sampling/sampling.md)). Parameter:
`temperature`. Owns **three** device buffers (ratios, numbers, mass).

### `MaxwellSigmaGenerator`

Draws from an isotropic Gaussian whose standard deviation is **given directly**
rather than derived from temperature and mass:

```cpp
velocity = (sigma > 0.0f) ? sample_normal_vector(engine, sigma) + bulk_velocity
                          : bulk_velocity;
```

When `sigma <= 0` the Gaussian collapses and every particle gets exactly the bulk
drift. This is how it differs from `MaxwellBoltzmannGenerator`: the spread is a
free parameter (`sigma`), so `_temperature` here is stored and validated (must be
finite and non-negative) but **does not** influence the sampled velocity. The
code states **no** relation between `sigma` and `temperature` — the two are
independent inputs, and nothing in this leaf converts one to the other.
Parameter: `sigma`. Owns two device buffers.

## Species selection

Species come from a weighted draw over the parallel `_species_ratios` (selection
weights) and `_species_numbers` (the species-id values). The draw needs the weights
**normalized** to sum to 1: both `sample_weighted_choice` and `sample_weighted_index`
walk a cumulative sum against `u ∈ [0, 1)`, so the first bucket whose cumulative
weight reaches 1 wins every draw otherwise (weights `{70, 30}` — or even `{1, 1}` —
always select index 0). `MaxwellBoltzmannGenerator::Builder::build()` rescales the
staged weights for you, so a caller may pass raw population counts or percentages
(e.g. `{70, 30}` builds as `{0.7, 0.3}`); the other leaves still expect weights that
already sum to 1. The draw uses the [sampling](../sampling/sampling.md) helpers, and
the leaves split into two conventions:

| Leaf | species draw |
|---|---|
| `UniformGenerator`, `JitteringGenerator`, `MaxwellSigmaGenerator` | `sample_weighted_choice(ratios, numbers, count, engine)` — returns `numbers[chosen]` (the id) directly |
| `MaxwellBoltzmannGenerator` | `sample_weighted_index(ratios, count, engine)` — returns the array **index**, then reads `numbers[index]` (id) and `masses[index]` (mass) so the id and its mass stay paired |

`MaxwellBoltzmannGenerator::Builder` resolves the per-species mass array two ways:
`with_species_mass` gives it explicitly (one entry per selectable species,
indexed like the ratios), or `with_material_dictionary` caches a
mass-by-material-id table so `build()` looks up `masses[k] = material_mass[numbers[k]]`.
If both are given, the explicit masses win; a species id outside the dictionary's
range throws at `build()`.

## Seeding and reproducibility

Each leaf carries an `unsigned int _seed`, and each builder exposes a `with_seed`
setter. When unset it falls back to `atlas::DEFAULT_UNSIGNED_INT_SEED` (`= 0`, in
[`random/seed.h`](../random/random.md)), so an unconfigured run is still
deterministic. `build()` also resets the builder's seed to that default after
each build.

Per-particle streams are derived by **hashing the absolute slot index with the
seed, then constructing one engine per slot** — not by materializing an engine
from the raw index and not by pure hashed draws. Inside every leaf's kernel:

```cpp
const std::size_t index = offset + i;
const std::uint64_t key = atlas::shuffle_key(static_cast<int>(index),
                                             static_cast<std::uint64_t>(seed));
atlas::default_random_engine engine(static_cast<unsigned int>(key));
```

`shuffle_key` is the SplitMix64 finalizer (see
[sampling](../sampling/sampling.md)); folding the absolute slot index with the
seed gives each particle a decorrelated, reproducible stream regardless of launch
order. The species draw and the velocity draw then advance that **same** engine,
so the several correlated quantities a particle needs come from one stream. The
engine is a Park-Miller LCG whose state is never zero, so a key that hashes to
zero is safe (see [random](../random/random.md)). Because both the TBB and CUDA
backends implement this engine identically, a CPU reproduction draws the same
stream as the GPU run.

## Emission flow — pairing with a source

`System::emit()` (`src/atlas/system/system.cu`) loops over the paired
`(source, generator)` lists on the host. For each pair it appends at the current
live particle `count`:

```cpp
const int spawned = _sources[i]->spawn(positions, count);       // writes positions [count, count+spawned)
if (spawned <= 0) continue;
static_cast<void>(_generators[i]->generate(velocities, species, // fills the SAME range
                                           count,
                                           static_cast<std::size_t>(spawned)));
count += static_cast<std::size_t>(spawned);
```

The source's `spawned` count fixes the slot range, so the generator's own return
value is discarded here (cast to `void`) — the two operate on exactly the same
`[count, count + spawned)` window. `System::Builder::with_emitter(source, generator)`
adds pairs, and the builder rejects a build where a source has no paired
generator or either is null. Each `spawn`/`generate` launches its own device
kernel; the host-side `HostVariant` dispatch is one visit per pair per step.

## Adding a leaf

Mirrors the collider/source recipe, using `HostVariant` because the leaves own
device buffers:

1. Write a move-only leaf satisfying `ConceptGenerator` (own its
   `DeviceBuffer<float>` species tables, implement `generate` and
   `set_bulk_velocity`, add a validating `Builder` with `with_seed` falling back
   to `DEFAULT_UNSIGNED_INT_SEED`). Follow the shared kernel shape: null/zero
   guards, `capacity`/`writable` clamping, `shuffle_key`-seeded per-slot engine.
2. Add its value to `GeneratorType` (keep the enumerator order in lockstep with
   the `HostVariantCase` list).
3. Add the union member in `Generator`, a matching `explicit Generator(NewLeaf)`
   payload constructor, and a `HostVariantCase<GeneratorType::new_leaf, &Generator::new_leaf>`
   to `GeneratorVariant`.
4. Add `static_assert(ConceptGenerator<NewLeaf>);` in `generator.h`.

Because the leaves own device buffers, `HostVariant` is the correct umbrella here
(contrast `Collider`/`Sink`, whose trivially-copyable leaves use `DeviceVariant`).
