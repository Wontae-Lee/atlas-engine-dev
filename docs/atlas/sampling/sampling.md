# Sampling

`sampling` is a header-only kit of free functions that turn randomness into the
concrete quantities the pipeline needs: a species id, a velocity, a scatter
direction, a collision partner, a lattice of emission seeds. It has no state and
no types of its own — every function is `ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE`, so
it inlines into the device lambdas of the modules that call it. It sits at the
producing ends of the step pipeline: the **generators** (emit) draw species and
velocities, the **sources** (emit) count lattice points, the **DSMC solver**
(solve) picks partners and accepts collisions, and the **isothermal collider**
(collide, inside advect/remove) picks diffuse-reflection directions. Two families
live here: engine-driven draws that advance a `default_random_engine`, and
*stateless hashed* draws that derive a reproducible value from an `(index, seed)`
or a geometric seed with no per-thread engine object.

## Files

| File | Role |
|---|---|
| `include/atlas/sampling/sampling.h` | The entire module: hash primitive, Box-Muller normals, vector/direction samplers, lattice-axis counter, hashed and weighted draws. |

There is no `src/atlas/sampling/`; everything is inline in the header. The mixing
constants and salts the functions reference (`SHUFFLE_HASH_*`, `RANDOM_HASH_*`,
`DSMC_COLLISION_*_SALT`) live in `include/atlas/random/seed.h`, not here.

## The primitives and their contracts

### `shuffle_key` — the stateless hash

```cpp
std::uint64_t shuffle_key(int index, std::uint64_t seed) noexcept;
```

The SplitMix64 finalizer. It offsets `index` by `seed` and the golden-ratio
constant `SHUFFLE_HASH_INDEX_OFFSET`, then applies two multiply/xor-shift
avalanche rounds and a final xor-shift. Branch-free, allocation-free, host and
device. This is the foundation under `sample_hashed_unit_interval(int, …)` and
`sample_hashed_index`, and the generators call it directly to seed a per-particle
engine. `index` is reinterpreted as unsigned, so a negative index wraps rather
than being rejected — the caller is trusted to pass a valid slot or cell id.

The generators use it as: hash `(particle_index, seed)` → build a
`default_random_engine` from the key → draw species and velocity. That is why
`shuffle_key` is public and not buried inside the hashed-interval helpers.

### Box-Muller normals

```cpp
void  generate_standard_normal_pair(default_random_engine& engine, float& first, float& second);
float generate_standard_normal(default_random_engine& engine);           // see "Not implemented"
Float3 sample_normal_vector(default_random_engine& engine, float sigma);
```

`generate_standard_normal_pair` is the workhorse: two uniforms in → two
independent `N(0,1)` out. It clamps `u1` up to `eps` before `log` so a drawn zero
cannot give `log(0) = -inf`, and takes the radius through `sqrt_nonnegative` as a
second guard. `sample_normal_vector` calls it twice for three normals (discarding
the fourth) and scales each by `sigma`, producing an isotropic Gaussian velocity —
the Maxwell-Boltzmann / Maxwell-sigma generators use it as `sample_normal_vector +
bulk`.

### Uniform vector and lattice counting

```cpp
Float3 sample_uniform_vector(default_random_engine& engine, float min_value, float max_value);
int    sample_axis_count(float lower, float upper, float spacing) noexcept;
```

`sample_uniform_vector` draws a point in the box `[min,max)^3`; the uniform and
jittering generators use it to scatter velocities and positions inside a cell.
`sample_axis_count` returns `floor((upper - lower) / spacing) + 1`, the number of
evenly spaced points along one axis including both endpoints; it returns `0` on
non-finite input, `spacing <= 0`, or `upper < lower`, and `1` for a zero-length
extent. The volume and surface sources call it once per axis to size their
emission lattice.

### Direction samplers

```cpp
void   build_orthonormal_basis(const Float3& n, Float3& t, Float3& b);          // see "Not implemented"
Float3 sample_uniform_hemisphere(const Float3& n, float u1, float u2);
Float3 sample_cosine_hemisphere(const Float3& n, float u1, float u2);
Float3 sample_random_unit_vector(default_random_engine& engine) noexcept;        // see "Not implemented"
Float3 sample_directional_unit_vector(const Float3& in, float alpha,
                                      default_random_engine& engine) noexcept;   // see "Not implemented"
```

The two hemisphere samplers take their uniforms *by value* rather than drawing
them, precisely so the caller can source them either from an engine or from a
stateless hash — the isothermal collider feeds them hashed draws. They reconstruct
the direction in `n`'s tangent frame via `spherical_direction`.
`sample_uniform_hemisphere` is solid-angle uniform (`cos θ = 1 - u1`);
`sample_cosine_hemisphere` is Lambertian (`cos θ = sqrt(1 - u2)`). Note the
warning baked into both: the `(u1, u2)` roles are *swapped* between them — `u1` is
the azimuth for the cosine variant and the polar angle for the uniform variant.

### Hashed draws

```cpp
float sample_hashed_unit_interval(const Float3& seed, float salt) noexcept;   // sine hash
float sample_hashed_unit_interval(int index, std::uint64_t seed) noexcept;    // shuffle_key hash
int   sample_hashed_index(int index, int upper_bound, std::uint64_t seed) noexcept;
```

Two overloads of `sample_hashed_unit_interval` with different quality. The
`Float3`/sine overload is the GLSL `fract(sin(dot(seed,k)+salt)·scale)` trick — a
*low-quality* hash whose own doc warns it has visible structure and is used only
where correlations are harmless (surface-scatter jitter in the isothermal
collider). The `(int, seed)` overload runs `shuffle_key` and keeps the top 53 bits
for a full-mantissa `[0,1)` draw; it feeds statistically critical tests, chiefly
the DSMC collision-acceptance probability (`dsmc_solver.cu`) and scatter
(`dsmc_scatter.h`). `sample_hashed_index` reduces `shuffle_key` modulo
`upper_bound` to pick a DSMC collision partner within a cell (with the documented,
negligible modulo bias); it returns `0` when `upper_bound <= 0`.

### Weighted draws

```cpp
int         sample_weighted_index (const float* weights, int count, default_random_engine& engine) noexcept;
std::size_t sample_weighted_choice(const float* weights, const float* values, int count,
                                   default_random_engine& engine) noexcept;
```

Inverse-CDF / roulette selection over a normalized weight table. `_index` returns
the chosen bucket; `_choice` returns `values[chosen]` truncated to `std::size_t`.
The generators use these to turn species-population ratios into a species id: the
Maxwell-Boltzmann leaf takes the *index* (it looks up sigma from a parallel table
by that index), while the uniform, jittering, and Maxwell-sigma leaves take the
*choice* (the parallel `values` table holds the species ids directly). Both return
`0` when `count <= 0`; when the weights sum to less than 1, a draw past the last
threshold falls through to `count - 1` — a deliberate in-range clamp, not an error.

## Why the shape is what it is

- **No leaf, no umbrella.** Unlike the tagged-union modules, sampling is a flat
  bag of free functions. There is nothing to dispatch on: each caller knows the
  exact distribution it wants at the call site.
- **Everything is `ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE` in the header.** These are
  called from inside device lambdas in `.cu` files. Header-inline device code is
  the only form that inlines across that boundary, so there is no `.cu`.
- **Stateless vs. engine-driven is a deliberate split.** The hashed helpers exist
  so a device thread can produce reproducible randomness from `(index, seed)`
  without materializing and carrying a `default_random_engine`; the engine-driven
  helpers exist for the generators, which build one engine per particle from a
  `shuffle_key` and then draw several correlated quantities from it.
- **Constants live in `random/seed.h`, not here.** The functions are pure logic;
  the magic numbers (avalanche shifts/multipliers, sine-hash phase coefficients,
  the `1/2^53` scale, the per-purpose salts) are named constants elsewhere so the
  same seed policy is shared across modules.

## Deliberately absent

- **No engine object in the hashed path.** By design — the point of the hashed
  helpers is exactly to avoid per-thread engine state (see above).
- **The sine-hash overload is knowingly low quality.** Its own `@warning` says so.
  It is kept for cheap, correlation-tolerant jitter, not replaced with the
  high-quality shuffle-hash there, because the visible structure is harmless for
  surface-scatter perturbation.
- **`sample_hashed_index`'s modulo bias is accepted.** Documented as negligible for
  the small per-cell partner counts DSMC uses; not corrected with rejection
  sampling.
- **Weighted draws are not bounds-checked on the device** and assume normalized,
  non-negative weights; the `< 1` sum is clamped to the last index rather than
  reported. A deliberate keep-in-range choice consistent with the engine's
  large-domain, skip-fine-detail philosophy.

## Not implemented

Every function was grepped across `include`, `src`, `tests`, and `benchmarks`
(excluding this doc and the type's own definition). Four public functions have no
caller anywhere — no engine code, no test, no benchmark reaches them. They are
declared and documented but dead. The rest are all reached (see the callers noted
inline above). Nothing here should be deleted on this evidence alone; recorded for
a human to decide.

| What | Where | Evidence | Verdict |
|---|---|---|---|
| `generate_standard_normal(engine)` | `sampling.h:84` | Only mentions outside its own body are `@see`/doc references (`default_random_engine.h:27`). Callers draw normals through `generate_standard_normal_pair` or `sample_normal_vector` instead. | Declared, never called (dead). |
| `build_orthonormal_basis(n,t,b)` | `sampling.h:158` | The only occurrence of the name in the entire tree is its own definition. The underlying `orthonormal_basis` in `math/vector/float3.h` is used directly by other code; this safe-fallback wrapper is not. | Declared, never called (dead). |
| `sample_random_unit_vector(engine)` | `sampling.h:231` | Only occurrence of the name is its own definition. | Declared, never called (dead). |
| `sample_directional_unit_vector(in,alpha,engine)` | `sampling.h:262` | Only occurrence of the name is its own definition. | Declared, never called (dead). |

Notes:

- `generate_standard_normal_pair` is **not** dead — it has no external caller of
  its own but is reached internally through `generate_standard_normal` (dead) and
  `sample_normal_vector` (live, used by the Maxwell generators), so the live path
  keeps it exercised.
- The four dead functions are plausibly *extension points* (a full sphere sampler,
  a power-cosine lobe, a reusable basis builder) staged for a scatter or emission
  model not yet wired in — but nothing in the current engine, tests, or benchmarks
  produces or consumes them. This is a fact recorded, not a deletion proposal.

## Extending

- **Add a new sampler.** Write one more `ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE` free
  function in `sampling.h` in the `atlas` namespace, matching the two calling
  conventions already here: take a `default_random_engine&` for a stateful draw,
  or take pre-drawn `u1, u2` (and/or an `(index, seed)`) for a stateless one so a
  device thread can call it without an engine. Document units, ranges, what a
  degenerate argument does, and whether it runs host/device — the coding-style
  rules apply because this header is under `include/atlas/`.
- **Add a hash stream.** Do not hard-code new constants or salts in a function
  body; add a named constant to `include/atlas/random/seed.h` (as the existing
  `SHUFFLE_HASH_*`, `RANDOM_HASH_*`, and `DSMC_COLLISION_*_SALT` are) and reference
  it, so the seed policy stays in one place.
- **No `validate()` or `static_assert` guards here.** This module has no builder
  and no umbrella type, so there is no schema check to satisfy. Correctness rests
  on the in-range/short-circuit contracts each function documents (`count <= 0 →
  0`, `upper_bound <= 0 → 0`, non-finite lattice input → `0`); preserve that style
  rather than adding throwing validation, which does not exist on this device-side
  path.
