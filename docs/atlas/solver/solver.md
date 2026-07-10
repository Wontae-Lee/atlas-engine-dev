# Solver

A solver advances one **particle-interaction physics** over the fluid, in place,
once per `System` step. Today there is exactly one: `DsmcSolver`, the Direct
Simulation Monte Carlo collision solver. Unlike the engine's trivially-copyable
leaves (`Collider`, `Material`, `DsmcKernel`), a solver is heavyweight — it owns
device scratch buffers — so the umbrella `Solver` is **not** a tagged union: it
is an abstract base with virtual dispatch, and the `System` holds a list of
`SolverHostPtr` and calls `solve` on each. `SolverType` is a runtime tag so a
caller can recover the concrete kind without RTTI.

The DSMC collision *models* — the cross section and the scattering law — do
follow the tagged-union leaf pattern, one level down: `DsmcKernel` is a
`DeviceVariant` over three stateless kernel leaves (`HardSphereKernel`,
`VariableHardSphereKernel`, `VariableSoftSphereKernel`), captured by value into
the collision device lambda.

## Files

| File | Role |
|---|---|
| `include/atlas/solver/solver.h` | `Solver` abstract base (`type` + `solve`), `SolverHostPtr` / `SolverDevicePtr` |
| `include/atlas/solver/solver_type.h` | `enum class SolverType : int { dsmc }` |
| `include/atlas/solver/dsmc/dsmc_solver.h` | `DsmcSolver` leaf + nested `Builder`, `DsmcSolverHostPtr` |
| `src/atlas/solver/dsmc/dsmc_solver.cu` | The NTC scheme: the two `solve` passes + `flatten_candidates` |
| `include/atlas/solver/dsmc/kernel/dsmc_kernel.h` | `ConceptDsmcKernel`, the `DsmcKernel` umbrella (`DeviceVariant`), visitors, `DsmcKernelVariant` |
| `include/atlas/solver/dsmc/kernel/dsmc_kernel_type.h` | `enum class DsmcKernelType : int { hard_sphere, variable_hard_sphere, variable_soft_sphere }` |
| `include/atlas/solver/dsmc/kernel/dsmc_scatter.h` | `dsmc_scatter` — the shared centre-of-mass elastic scatter (engine-driven) |
| `include/atlas/solver/dsmc/kernel/hard_sphere_kernel.h` | `HardSphereKernel` leaf — `pi d^2`, speed-independent, isotropic |
| `include/atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h` | `VariableHardSphereKernel` leaf — VHS cross section, isotropic |
| `include/atlas/solver/dsmc/kernel/variable_soft_sphere_kernel.h` | `VariableSoftSphereKernel` leaf — VHS cross section, anisotropic (VSS) |

## `Solver` and `SolverType`

`Solver` is an interface only: it stores no state, its special members are
`= default`, and it declares two pure-virtual operations.

```cpp
class Solver {
public:
    virtual SolverType type() const noexcept = 0;
    virtual void solve(Fluid&, Universe&, const SpatialHashingSearcherView&,
                       int index, float dt) = 0;
};

using SolverHostPtr = atlas::host_shared_ptr<Solver>;
```

- **`type()`** returns the `SolverType` enumerator (only `dsmc` today), so the
  System can branch on the dynamic kind without RTTI. The underlying type is
  fixed to `int` so the tag round-trips through serialization and device-side
  comparison.
- **`solve(fluid, universe, searcher_view, index, dt)`** reads the current
  particle state through views gathered on the host and mutates the fluid (only
  velocities) via device kernels. It is called **once per System step for each
  registered solver**.

### Several solvers sharing one grid

The `index` argument is the solver's position in the System's solver list, and
it is how multiple solvers coexist on **one** grid. The codec
(`docs/atlas/codec/codec.md`) writes a per-cell owning-solver index into
`UniverseAllocatedSolverState`; each solver then processes only the cells whose
`allocated_solver[cell] == index`. A **null** ownership buffer is the
single-solver case — every solver owns every cell. The free helper
`owns_cell(allocated_solver, cell, index)` in `dsmc_solver.cu` encodes exactly
that rule, and both `solve` passes gate on it.

## `DsmcSolver`

The only leaf. It is host-owned and dispatched virtually (the scratch
`DeviceBuffer` members make it move/host-owning rather than device-capturable),
and it carries just five configuration/state members:

| Member | Default | Meaning |
|---|---|---|
| `_majorant_sample_pairs` | `8` | Random pairs sampled to bound a large cell's majorant (`>= 1`). |
| `_majorant_exhaustive_limit` | `5` | Occupancy below which the majorant is scanned over all pairs (`>= 2`). |
| `_kernel` | hard sphere | The active `DsmcKernel` (cross section + scatter). |
| `_collision_seed` | `0` | Monotonic per-step stream base; incremented once per `solve`. |
| `_candidate_offsets` / `_candidate_cells` / `_owned_candidate_counts` / `_candidate_total` | — | Device scratch reused across steps by `flatten_candidates`. |

Construct it directly (`DsmcSolver(kernel_type, sample_pairs, exhaustive_limit)`)
or through the fluent `DsmcSolver::Builder`
(`with_kernel_type`, `with_majorant_sample_pairs`,
`with_majorant_exhaustive_limit`, then `build()` or `make_host_shared()`).
`Builder::validate` throws `std::runtime_error` if `majorant_sample_pairs < 1` or
`majorant_exhaustive_limit < 2` — a cell needs two particles before a pair exists
at all, so an exhaustive scan below that bound would sample nothing. The direct
constructor trusts its inputs and does not validate.

`solve` first gathers `FluidDsmcView` and `UniverseDsmcView` on the host and
**no-ops with a warning** when the fluid or universe is missing a needed state,
the searcher has not classified the particles, or the fluid carries no material
dictionary. It then **returns silently** when `dt <= 0`, the fluid is empty, the
material table is empty, or no candidate pairs were scheduled. Species indices
are read unchecked — `System::Builder` has already clamped every generator's
species id to the dictionary length.

## The NTC scheme

`DsmcSolver::solve` implements the **No-Time-Counter (NTC)** collision scheme in
two device passes. The governing quantity per cell is `sigma * g` (cross section
times relative speed) and its running upper bound `(sigma*g)_max`, the
*majorant*.

### Pass 1 — estimate the majorant and the candidate count

One thread per cell. For each **owned** cell with `count >= 2` particles (the
`[begin, end)` range comes from `searcher_view.cell_start/cell_end`):

- **Estimate `(sigma*g)_max`.** If `count < majorant_exhaustive_limit`, scan
  **every** unordered pair exactly (the double loop over local slots). Otherwise
  sample `majorant_sample_pairs` hashed random pairs. Either way,
  `accumulate_majorant` evaluates `kernel.sigma_g(...)` for the pair and keeps
  the running maxima of both the squared relative speed and `sigma * g`.
- **Persist the majorant.** The cell's stored `max_sigma_g` is read first and
  only ever **raised** by the sampled estimate; it is never lowered. Because the
  bound persists across steps, a rare fast pair keeps protecting the acceptance
  test on later steps too, and an under-estimate here costs *efficiency* (fewer
  candidates drawn), never *correctness*. `max_relative_speed` is written as
  `sqrt(max_relative_squared)` for downstream diagnostics.
- **Candidate count.** The expected number of candidate pairs is

  ```
  candidate_count = C(n,2) * (sigma*g)_max * W * dt / V
  ```

  with `C(n,2) = count*(count-1)/2`, `W` the fluid `statistical_weight`, `dt` the
  sub-step, and `V` the `cell_volume`. It is written into `collision_count[cell]`
  as `floor(expected)` — the fractional remainder is simply dropped rather than
  carried as per-cell state. Cells with `count < 2`, a non-positive majorant, or
  a non-positive cell volume get `collision_count = 0`.

### Pass 2 — one work item per candidate

Rather than launch one thread per cell (which would idle whole warps when cell
occupancy varies wildly), the per-cell counts are **flattened** into a single
candidate list by `flatten_candidates`, then one thread runs per candidate.

**`flatten_candidates(universe_view, index)`** (public only because it launches
an extended `__host__ __device__` lambda, which nvcc forbids inside a
private/protected member):

1. If several solvers share the grid (`allocated_solver != nullptr`), the other
   solvers' counts are first **masked to 0** into the scratch
   `_owned_candidate_counts` — the caller's `collision_count` is left intact
   because the cell's next owner still needs it.
2. `exclusive_scan` the (possibly masked) counts into `_candidate_offsets`.
3. Compute the grand total on the device (one thread:
   `offsets[last] + counts[last]`) and copy back only that single scalar, so the
   per-cell counts never round-trip to the host.
4. Fill `_candidate_cells` so entry `w` names the cell owning the `w`-th
   candidate, found by a **binary search** (`upper_bound`-style) over the
   offsets. Every scratch buffer grows as needed and is reused across steps.

The collision kernel then runs one thread per flat candidate. For candidate
`work_index` in cell `cell`, its **position within the cell** is
`collision = work_index - candidate_offsets[cell]`, so the hashed stream matches
exactly what a per-cell loop would have drawn — this is why pass 1's sampling and
pass 2 agree. Each candidate:

- **picks its pair** with `sample_distinct_pair` (below),
- **recomputes** `sigma * g` for that pair's actual relative velocity,
- **self-corrects the majorant**: a candidate whose `sigma * g` exceeds the
  current bound raises `max_sigma_g[cell]` for the rest of this step and the
  next,
- **accepts** with probability `sigma*g / (sigma*g)_max` (a candidate at or above
  the bound is accepted with probability 1). The acceptance draw is
  `sample_hashed_unit_interval(cell, stream + DSMC_COLLISION_ACCEPT_SALT)` — a
  high-quality `shuffle_key` draw keyed on **indices**, not on physical state, so
  it carries none of the bias described for the old scatter hash.
- On acceptance it seeds a per-collision engine and calls the kernel's scatter,
  then writes both velocities back.

### `sigma_g` takes the *squared* speed

`DsmcKernel::sigma_g(materials, lhs_species, rhs_species, relative_speed_squared)`
takes the **squared** relative speed on purpose: the caller already has
`(v_lhs - v_rhs).length_squared()`, so passing the square avoids a redundant
`sqrt` at every call site. Internally it short-circuits to `0` when the squared
speed is non-positive (no relative motion → no collision flux), then computes
`relative_speed = sqrt(...)` once and returns `cross_section * relative_speed`
(m^3/s). Materials are indexed by species id without a bounds check.

### Partner selection and the per-cell stream

`sample_distinct_pair(lhs, rhs, cell, count, stream)` draws two **distinct** local
slots in a cell of `count` particles: a first slot in `[0, count)` and a second in
`[0, count - 1)`, each a `sample_hashed_index` draw salted with
`DSMC_COLLISION_LHS_SALT` / `DSMC_COLLISION_RHS_SALT`; the second is then bumped up
by one whenever it lands on or after the first. That is the standard trick for
sampling an unordered distinct pair uniformly without rejection. It is stateless,
so the same `(cell, stream)` always yields the same pair — the property that lets
pass 1 and pass 2 agree.

The per-collision `stream` base is

```
stream = collision_seed
       + cell * DSMC_CELL_STREAM_MULTIPLIER
       + collision_index
```

where `collision_seed = _collision_seed++` is fixed for the whole step and
`DSMC_CELL_STREAM_MULTIPLIER` (a golden-ratio odd constant) gives each cell a
well-separated stream so neighbouring cells never draw correlated partners. The
partner, acceptance, and scatter draws all derive from this same `stream` base,
each with its own salt. See `docs/atlas/random/random.md` and
`docs/atlas/sampling/sampling.md` for the seed constants and the hashed-draw
helpers this consumes.

## The kernels

`DsmcKernel` is a trivially-copyable `DeviceVariant` over three leaves; `type` is
the public discriminator. Every leaf satisfies `ConceptDsmcKernel`:

```cpp
template <typename K>
concept ConceptDsmcKernel = requires(const K kernel, Float3 velocity,
                                     const Material material, float relative_speed,
                                     default_random_engine& engine) {
    { K::cross_section(material, material, relative_speed) } -> std::same_as<float>;
    { kernel(velocity, velocity, material, material, engine) } -> std::same_as<void>;
};
```

- a **static** `cross_section(lhs, rhs, relative_speed)` returning m^2, and
- a **const** call operator that scatters the pair in place, drawing its variates
  from a caller-supplied `atlas::default_random_engine&`.

Three `static_assert(ConceptDsmcKernel<...>)` lines check every leaf conforms. The
umbrella exposes `cross_section`, the `sigma_g` convenience wrapper, and the
scatter `operator()`; each dispatches to the active leaf through `DsmcKernelVariant`
(with the `DsmcCrossSection` / `DsmcCollide` visitor functors — functors, not
lambdas, because nvcc forbids an extended `__host__ __device__` lambda in class
scope). `cross_section` falls back to `0` if the tag matches no arm, so an unset
kernel simply collides nothing.

**The leaves are stateless PODs.** They hold no parameters of their own — *every*
physical parameter comes from `Material` (`docs/atlas/material/material.md`). That
is what keeps `DsmcKernel` trivially copyable so a `DsmcSolver` can capture it by
value into the device lambda; correspondingly the RNG **engine is passed by
reference, not owned** by the leaf.

### `HardSphereKernel` — `pi * d^2`, speed-independent

Each species is a rigid sphere of fixed diameter, so the pair cross section is
`pi * d^2` with `d = (d_ref_lhs + d_ref_rhs) / 2` the arithmetic **mean reference
diameter**. The `relative_speed` argument is ignored — the defining property of
the hard-sphere model. Returns `0` if the mean diameter is non-positive.
Scattering is isotropic (`dsmc_scatter` with `alpha = 1`).

### `VariableHardSphereKernel` — VHS cross section, isotropic scatter

Refines the hard-sphere model so the cross section falls off with relative speed
as a power law set by the viscosity index `omega`, reproducing a realistic
temperature-dependent viscosity. Every pair property is an arithmetic mean of the
two species' values. The cross section is

```
sigma = pi * d_ref^2 * (2 k T_ref / (m_r g^2))^(omega - 1/2) / Gamma(5/2 - omega)
```

with `d_ref`, `T_ref`, `omega` the pair means, `m_r` the reduced mass, `g` the
relative speed, and `k` Boltzmann's constant. The exponent is exactly
`viscosity_index - 0.5` (i.e. `omega - 1/2`). Each intermediate is guarded so a
degenerate species yields `0` rather than a NaN; in particular it returns **0**:

- when either mass, the mass sum, or the relative speed is non-positive (so at
  **zero relative speed** `sigma = 0`),
- when the mean reference diameter or temperature is non-positive,
- when the **Gamma argument `2.5 - omega` is non-positive**, i.e. `omega >= 2.5`,
- or when `Gamma(5/2 - omega)` or the guarded `m_r g^2` product is non-positive.

The reduced mass is evaluated as `m_l * (m_r / (m_l + m_r))` — **divide before
multiply** — never as `m_l * m_r / (m_l + m_r)`: a molecular mass is ~1e-26 kg, so
the bare product underflows float to zero and the expression would return `inf`.
`std::tgamma` is evaluated in double and narrowed back. Scattering is isotropic
(`alpha = 1`), like hard spheres; only the cross section differs.

### `VariableSoftSphereKernel` — VHS cross section, anisotropic scatter

The cross section is **identical** to VHS (it forwards to
`VariableHardSphereKernel::cross_section`). VSS and VHS differ only in the
*scattering angle*: VSS uses the soft-sphere angular law with `alpha` the
arithmetic **mean of the two species' scattering parameters** (a mean of `1`
reproduces isotropic VHS scatter), letting a model match both viscosity and
diffusion coefficients.

## `dsmc_scatter` — the shared elastic scatter

`dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, scattering_parameter, engine)`
is the single scatter routine all three leaves call; only the `alpha` they pass
differs. It works in the pair's **centre-of-mass frame**: the centre-of-mass
velocity and the relative *speed* are conserved, and only the **direction** of the
relative velocity is redrawn — which conserves both momentum and kinetic energy
exactly (the tests verify momentum, energy, relative speed, and the centre-of-mass
velocity are all preserved). The physics:

- **Deflection cosine** from the VSS law `cos(chi) = 2 * u1^(1/alpha) - 1`, which
  collapses to the isotropic `cos(chi) = 2*u1 - 1` at `alpha == 1` (a dedicated
  branch that skips the `pow`). The mean over many draws is
  `E[cos chi] = (alpha - 1) / (alpha + 1)` — 0 at `alpha = 1`.
- **Azimuth** `phi = 2*pi*u2`, uniform.
- The relative velocity is rotated to `spherical_direction(axis, cos_chi, phi)`,
  its magnitude preserved, then redistributed about the conserved centre of mass
  weighting each partner by the *other's* mass.

**It is a no-op** — and consumes no variates — when either mass, the mass sum, the
scattering parameter, or the relative speed is non-positive; a degenerate pair has
no well-defined scatter direction.

### The engine, and why it replaced a hash

The two uniforms `u1`, `u2` are drawn from the trailing
`atlas::default_random_engine& engine` via `uniform_real_distribution<float>`, so a
successful scatter **advances the engine exactly twice** (the "consumes exactly
two variates" and "no-op leaves the engine untouched" behaviours are both tested).

The engine is a recent, deliberate change. The previous implementation keyed a
low-quality sine hash on the pair's **own velocities**, which made `cos(chi)` a
*deterministic function of the pre-collision state* — two particles meeting at the
same relative velocity always deflected identically — and inherited the sine hash's
sensitivity to input magnitude. Sourcing the variates from a seeded stream instead
makes the angular law independent of the collision's velocities (a test scatters a
slow and a fast pair under the same seed and asserts equal deflection cosines).

The solver seeds that engine per collision in
**`make_scatter_engine(cell, stream)`**: it folds `DSMC_COLLISION_SCATTER_SALT`
into the collision's `(cell, stream)` through `shuffle_key` (the SplitMix64
finalizer), then folds the 64-bit key down to 32 bits (`key ^ (key >> 32)`) because
the engine's state is 32-bit. The SplitMix64 mix matters: seeding an LCG with merely
*adjacent* values (as `(cell, collision)` are) would emit correlated first draws,
so the key must be avalanched first. `default_random_engine::seed` maps a
zero-congruent fold onto its default seed, so no guard is needed.

**Reproducibility now comes from the seed, not from a hash of physical state.** A
replay of the same step reproduces the same scatter (seeded identically), while the
angle is decoupled from the pair's velocities. Note the contrast with the NTC
**acceptance** draw, which is *still* a hashed `(index, seed)` draw — being keyed on
indices rather than physical state, it never had this bias.

## Adding a kernel leaf

Mirrors the collider/material checklist:

1. Write a **stateless, trivially-copyable POD** leaf satisfying
   `ConceptDsmcKernel`: a static `cross_section(lhs, rhs, relative_speed)` and a
   const `operator()(lhs_velocity, rhs_velocity, lhs, rhs, engine)` that scatters
   (typically by forwarding to `dsmc_scatter` with an `alpha`). Take **all**
   physical parameters from `Material` — the leaf must own no fields.
2. Add its enumerator to `DsmcKernelType`.
3. Add the union member to `DsmcKernel` and a `DeviceVariantCase` binding the tag
   to that member in the `DsmcKernelVariant` alias.
4. Add `static_assert(ConceptDsmcKernel<NewLeaf>);`.

The leaf **must stay a stateless POD**: if it owned a resource or held state,
`DsmcKernel` would stop being trivially copyable, the `= default` special members
would become ill-formed, and it could no longer be captured by value into the
collision device lambda. This is exactly why the RNG engine is passed by reference
rather than owned. (Contrast `Solver` itself, which owns `DeviceBuffer`s and is
therefore a virtual-dispatch base, not a `DeviceVariant`.)
