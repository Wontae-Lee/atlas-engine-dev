# Random

The random module is Atlas's own, backend-agnostic pseudo-random plumbing. It is
not a step in the `System::update()` pipeline; it is a **primitive** the emit
phase (the particle generators) and the solve phase (the DSMC solver, the
colliders, the spatial hashing) build on. It exists so that a host (TBB) build
and a CUDA build draw the **identical random stream** — a case reproduced on the
CPU must not diverge from the GPU run it is meant to check. Everything here is
header-only and `ATLAS_ALL_DEVICE`, so each device thread constructs an engine
from its own index and draws locally, with no shared mutable state and no locking.

## Files

| File | Role |
|---|---|
| `include/atlas/random/default_random_engine.h` | `default_random_engine` — a Park-Miller minimal-standard LCG (`x = 48271·x mod 2^31-1`) |
| `include/atlas/random/uniform_real_distribution.h` | `uniform_real_distribution<T=float>` — maps an engine draw onto `[min, max)` |
| `include/atlas/random/seed.h` | Named `constexpr` constants for the engine's *stateless* hashing/mixing (sine-hash, SplitMix64, per-purpose salts, Morton bit-spread) |

There is no `src/atlas/random/` — the whole module is inline in headers, as
`ATLAS_ALL_DEVICE` device-reachable code must be.

## `default_random_engine`

A hand-written Park-Miller LCG, deliberately **not** aliased to
`thrust::default_random_engine`. It uses the same recurrence thrust does, but
implementing it locally is what lets both backends agree bit-for-bit — the TBB
backend never links Thrust at all.

```cpp
class default_random_engine final {
    using result_type = std::uint32_t;
    static constexpr result_type multiplier   = 48271u;
    static constexpr result_type modulus      = 2147483647u;   // 2^31 - 1, prime
    static constexpr result_type default_seed = 1u;

    default_random_engine() noexcept;                  // seeds with default_seed
    explicit default_random_engine(result_type s) noexcept;
    void seed(result_type s) noexcept;
    result_type operator()() noexcept;                 // advance, return new state
    void discard(unsigned long long count) noexcept;
    static constexpr result_type min() noexcept;       // 1
    static constexpr result_type max() noexcept;       // modulus - 1
};
```

Contracts that the signature cannot show:

- **The state is never zero.** `48271·0 mod m == 0`, so a zero state would emit
  zero forever. `seed()` reduces its argument modulo `m` and maps a zero result
  onto `default_seed == 1`. Callers that seed from a hash key (the generators do,
  via `shuffle_key`) are therefore safe even when the key hashes to zero. This is
  why `min()` is `1`, not `0`.
- **64-bit intermediate.** `48271 · state` overflows 32 bits, so `operator()`
  forms the product in `std::uint64_t` before the modulo. That is the only reason
  for the temporary.
- **Trivially copyable, every member `ATLAS_ALL_DEVICE`.** The engine is a
  value captured into device lambdas; each thread owns its own.

The engine is **produced** in exactly four places — the generators
`jittering_generator.cu:84`, `uniform_generator.cu:84`,
`maxwell_boltzmann_generator.cu:81`, `maxwell_sigma_generator.cu:79` — each
`default_random_engine engine(static_cast<unsigned int>(key))` where `key` is a
per-slot `shuffle_key(index, seed)`. It is **consumed** entirely through the
`sampling/` free functions (`generate_standard_normal_pair`,
`sample_uniform_vector`, `sample_normal_vector`, `sample_random_unit_vector`,
`sample_weighted_choice`, and so on), which take it by `default_random_engine&`
and advance it.

## `uniform_real_distribution<T = float>`

Maps one engine draw onto a canonical variate and rescales it to `[min, max)`.
Also implemented locally rather than aliased, for the same both-backends-agree
reason. `T` defaults to `float`, matching the single-precision device kernels.

```cpp
template <typename T = float>
class uniform_real_distribution final {
    uniform_real_distribution() noexcept;                    // [0, 1)
    uniform_real_distribution(T min_value, T max_value) noexcept;
    result_type operator()(default_random_engine& engine) const noexcept;
    result_type min() const noexcept;
    result_type max() const noexcept;
};
```

- **The interval stays half-open.** The engine yields `[1, m-1]`, so
  `(x - 1) / (m - 1 + 1)` lands in `[0, 1)`. But rounding the quotient to `float`
  can round the largest states up to exactly `1`; `operator()` nudges that case
  back to `1 - 2^-24` (`nextafter_below_one()`). Callers that take `log(u)` —
  `generate_standard_normal_pair`'s Box-Muller — rely on `u < 1`.
- **Ordering is the caller's job.** The two-argument constructor does not check
  `max >= min`; a reversed pair simply yields draws outside the nominal interval.
  Consistent with the no-defensive-checks house rule.

Every instantiation across the tree uses the **two-argument** constructor, almost
always `(0.0f, 1.0f)` to get `u ∈ [0, 1)` for a subsequent transform (Box-Muller,
inverse CDF, box sampling): `sampling.h:59, 109, 233, 266, 412`.

## `seed.h` — stateless hashing constants

Atlas prefers *stateless* device randomness: rather than thread an RNG object
across kernels, most collision/scatter/spatial code derives a value straight from
integer indices, cell ids, and a per-time-step "stream" through a hash. `seed.h`
is where every otherwise-opaque magic number for those hashes lives, so a run
(and a restart) reproduces. The values are pure named `constexpr`s — this file
declares no function — grouped into four families:

- **Sine-hash** (`RANDOM_HASH_PHASE_COEFF_{X,Y,Z}`, `RANDOM_HASH_VALUE_SCALE`,
  the `SALT_*` and `NORMAL_SCALE_FOR_MIX` decorrelators) — the GLSL
  `fract(sin(dot(seed,k))·scale)` one-liner, consumed by
  `sample_hashed_unit_interval(const Float3&, float)` in `sampling.h`.
- **SplitMix64 finalizer** (`SHUFFLE_HASH_*`) — consumed by `shuffle_key` in
  `sampling.h`, which every generator calls to build its per-slot key.
- **DSMC per-cell stream** (`DSMC_CELL_STREAM_MULTIPLIER`, the collision
  `*_SALT`s) — consumed by the DSMC solver / scatter kernel so neighboring cells
  draw uncorrelated partners.
- **Morton bit-spread** (`MORTON_EXPAND_BITS_*`) — consumed by the LBVH to
  interleave a 10-bit coordinate into a 30-bit Morton code.
- **Defaults** — `DEFAULT_UNSIGNED_INT_SEED` (the fallback seed all four
  generators' builders reset to) and the `UNIT_INTERVAL_SHIFT`/`SCALE` pair for
  the 53-bit double-exact draw.

Every constant in `seed.h` has at least one consumer outside the file; none is
orphaned.

## Deliberately absent

- **No `<random>`, no `thrust::random`.** The recurrence and the distribution are
  reimplemented by hand precisely so the CUDA and TBB backends produce the same
  stream. Aliasing a standard-library or Thrust engine would make the two builds
  diverge, defeating CPU-vs-GPU reproducibility.
- **Not cryptographic.** A minimal-standard LCG is fast and cheap to copy but weak;
  an accepted trade-off for Monte-Carlo particle sampling, stated in the header.
- **One distribution only.** `uniform_real_distribution` is the sole distribution
  type; every other shape (normal, unit vector, weighted choice) is a transform of
  a uniform draw living in `sampling/`, not a distribution class here.
- **Stateless-first.** The stateful engine is used only where a thread needs many
  correlated draws (the generators). Hot device paths hash indices directly through
  `seed.h`, carrying no RNG object — that is the whole reason `seed.h` exists.

## Not implemented

Nothing in this module is an orphaned type or an accepted-but-ignored parameter;
the gaps are unused corners of two otherwise-live `std`-shaped RNG interfaces.

| What | Where | Evidence | Kind |
|---|---|---|---|
| `default_random_engine::discard(unsigned long long)` | `default_random_engine.h:93` | A grep of `include src tests benchmarks` for `discard` finds no call on an engine anywhere (only unrelated matches: `float3` overloads, the `ATLAS_NODISCARD` macro, doc prose). | Declared, never called. Present for `UniformRandomBitGenerator` interface completeness, not by design need. |
| `default_random_engine::seed(result_type)` as a re-seed | `default_random_engine.h:64` | Its only caller is the engine's own seeding constructor (`default_random_engine.h:56`). No external `.seed(` / `->seed(` call exists. | Reached, but never used as a standalone re-seed — every producer constructs a fresh engine instead of re-seeding one. |
| `default_random_engine()` default (no-arg) constructor | `default_random_engine.h:47` | Grep finds only the definition line; every producer uses the explicit seeding constructor. | Declared, never called. Interface completeness. |
| `uniform_real_distribution::min()` / `max()` getters | `uniform_real_distribution.h:73, 79` | No `.min()` / `.max()` call on a distribution object anywhere in the tree (the only static `min()`/`max()` uses are `default_random_engine::{min,max}` inside `operator()`). | Declared, never called. Standard-distribution-shaped accessors with no consumer. |
| `uniform_real_distribution()` default (no-arg) constructor | `uniform_real_distribution.h:33` | Every instantiation (`sampling.h:59, 109, 233, 266, 412`) uses the two-argument constructor; no `uniform_real_distribution<...> name;` form exists. | Declared, never called. The two-arg `(0,1)` form is used even where `[0,1)` is wanted. |

None of these is dead by design flaw — they are the unexercised members of a
deliberately `std`-shaped interface. Recorded for a human to decide; nothing here
proposes removal.

## Extending

- **A different generator or period**: change `multiplier` / `modulus` /
  `default_seed` in `default_random_engine`, keeping the never-zero invariant
  (`min() == 1`) — the distribution derives its `range` from `max() - min() + 1`,
  so it follows automatically. Both backends pick up the change identically since
  there is only the one implementation.
- **A new distribution shape**: prefer a free function in `sampling/` that takes
  `default_random_engine&` and transforms a `uniform_real_distribution` draw
  (that is how normal, unit-vector, and weighted-choice sampling already work),
  rather than adding a distribution class here.
- **A new hashed-randomness site**: add its magic numbers to `seed.h` as named
  `constexpr`s next to their family and document each with a `@see` to the
  consuming function — do not inline a bare literal at the call site, which is the
  convention this file exists to enforce.
- There is no `validate()` or `static_assert` in this module. The one hard
  invariant — a non-zero state — is enforced at runtime by `seed()` mapping a
  zero-congruent seed onto `default_seed`; a mis-ordered distribution interval is
  not checked and is the caller's responsibility.
