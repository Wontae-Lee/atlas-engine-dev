# Universe

The universe module is the engine's **Eulerian background grid**: a uniform,
axis-aligned lattice of cubic cells covering the simulation domain, plus an
open-ended set of per-cell field arrays. It is the fixed-grid counterpart to the
Lagrangian particle set in the [fluid module](../fluid/fluid.md). The `Universe`
defines the cell geometry and owns the per-cell fields (temperature, DSMC
counters, forces, ...); solvers reach the field data on the device through a
trivially-copyable *view* of raw device pointers (`UniverseDsmcView`).

## Files

| File | Role |
|---|---|
| `include/atlas/universe/universe.h` | `Universe` (grid + fields) + `Universe::Builder`; `UniverseStateStore = TypeStore<UniverseState>`; `UniverseHostPtr` |
| `include/atlas/universe/universe_state.h` | `UniverseState` base + the eleven per-cell field leaves |
| `include/atlas/universe/universe_view.h` | `UniverseDsmcView` — device-capturable snapshot for the DSMC collision kernel |
| `src/atlas/universe/universe.cu` | `Universe` / `Builder` definitions, including the grid-size derivation |
| `src/atlas/universe/universe_state.cu` | The leaf constructors and `size` / `reset` overrides |

## The grid

The grid is cached host-side metadata; only the field data lives in device
memory. The constructor takes `[lower_corner, upper_corner]` and a `cell_size`
and derives the rest once:

- `cell_volume = cell_size^3` — the number-density denominator in the DSMC
  collision rate.
- `inverse_cell_size = 1 / cell_size` — turns world-to-cell scaling into a
  multiply.
- `grid_size = floor((upper - lower) / cell_size) + 1` per axis — the `+1` adds
  the cell covering the remainder, so a box whose extent is not an exact multiple
  of `cell_size` rounds *up* to a full covering cell and every non-degenerate box
  spans at least one cell (`tests/atlas/universe/universe_tests.cpp` pins the unit
  box to a 2×2×2 = 8-cell grid).
- `cell_count = grid_size.x * grid_size.y * grid_size.z`, guaranteed to fit `int`.

Prefer the fluent `Universe::Builder` (`with_lower_corner`, `with_upper_corner`,
`with_cell_size`, or `with_geometry` to fit a geometry's bounding box). Its
`build()` / `make_host_unique()` validate the configuration — `cell_size > 0`,
`upper_corner` strictly greater than `lower_corner` on every axis, and a positive
cell count that fits `int` — throwing `std::invalid_argument` otherwise.

```cpp
UniverseHostPtr universe = Universe::builder()
    .with_lower_corner(Float3(0, 0, 0))
    .with_upper_corner(Float3(2, 2, 2))
    .with_cell_size(1.0f)          // -> 3x3x3 = 27 cells
    .make_host_unique();
```

## State storage — the `TypeStore`

The per-cell fields are held type-erased in a `UniverseStateStore` =
`TypeStore<UniverseState>` (see `include/atlas/container/type_store.h`), keyed on
the concrete C++ leaf type: **at most one field per type**, and inserting a second
of the same type replaces the first. Each field is a `UniverseState` leaf owning
one `DeviceBuffer` sized to `cell_count` (the sized `DeviceBuffer` constructor
value-initializes, so a freshly emplaced field starts zero-filled). The base
fixes `size()` and `reset()` (device-side zero-fill); there is no `compact()` here
because the cell count is fixed.

```cpp
const auto cells = static_cast<std::size_t>(universe.cell_count());
universe.emplace_state<UniverseNumberParticleState>(cells);   // add a field
auto* np = universe.state<UniverseNumberParticleState>();      // lookup, or nullptr
bool has = universe.has_state<UniverseNumberParticleState>();
auto owned = universe.remove_state<UniverseNumberParticleState>();  // detach
```

Unlike a `Fluid`, a `Universe` seeds **no** fields at construction. `System`
provisions exactly the columns its configured solvers need, idempotently:
`System::initialize_states` always ensures `UniverseAllocatedSolverState`, and any
DSMC solver additionally pulls in the four DSMC fields
(`src/atlas/system/system.cu`, via `ensure_universe_state`).

## The per-cell fields (`UniverseState` leaves)

All eleven leaves exist; the table lists the field element type and the step-phase
that writes and reads each. The four fields marked **DSMC** plus `allocated_solver`
are the ones the collision view gathers.

| Leaf | Element | Written by | Read by |
|---|---|---|---|
| `UniverseNumberParticleState` (`number_particle`) | `float` | the **search** pass (spatial-hashing classify bins particles into cells) | the codec **allocate** pass and the DSMC **solve** pass (the pair count `N`) |
| `UniverseCollisionCountState` (`collision_count`) | `int` | the DSMC **solve** pass (the NTC candidate count / scheduling slot) | the DSMC **solve** pass (read back to flatten per-cell candidates into a flat work list) |
| `UniverseMaxRelativeSpeedState` (`max_relative_speed`) | `float` | the DSMC **solve** pass each step (diagnostic: largest sampled pair relative speed) | observation |
| `UniverseMaxSigmaGState` (`max_sigma_g`) | `float` | the DSMC **solve** pass — read, then **raised in place** when a larger `sigma*g` is sampled | the DSMC **solve** pass, as the NTC majorant `(sigma*g)_max` bounding candidates and the accept probability |
| `UniverseAllocatedSolverState` (`allocated_solver`) | `int` | the codec **allocate** pass (per-cell owning-solver index) | the DSMC **solve** pass (a solver processes a cell only when this equals its own index; a null buffer means "every solver owns every cell") |
| `UniverseTemperatureState` (`temperature`) | `float` (K) | populated externally / by diagnostics | the codec **allocate** pass |
| `UniverseBulkVelocityState` (`bulk_velocity`) | `Float3` (m/s) | *reserved* | mean-flow diagnostic |
| `UniverseFieldForceState` (`field_force`) | `Float3` | *reserved* | external body-force field |
| `UniverseGravityState` (`gravity`) | `Float3` | *reserved* | gravity, kept separate from `field_force` |
| `UniverseThermalEnergyState` (`thermal_energy`) | `float` (J) | *reserved* | per-cell internal energy |
| `UniverseKnudsenNumberState` (`knudsen_number`) | `float` | *reserved* | rarefaction diagnostic for continuum-vs-DSMC selection |

The five fields marked *reserved* are defined and zero-initialized, but no pass in
the current step pipeline writes or reads them. They are placeholders for planned
work — body forces, gravity, and a Knudsen-driven continuum-vs-DSMC selection —
not dead code. Treat them as the intended extension points they are: a new pass
that produces one of them needs no change to `Universe` itself.

### The DSMC majorant persists across steps

`max_sigma_g` is the key quantity of the DSMC No-Time-Counter scheme. The solver
reads the *previous* step's value from `universe_view.max_sigma_g[cell]` and only
raises it when it samples a larger `sigma*g` — it never zeroes the field between
steps. So the majorant **persists across steps** and converges upward, which is
exactly what bounds the candidate count and the `sigma*g / (sigma*g)_max` accept
probability. (`reset()` exists on the leaf, but the DSMC solver does not call it
per step.)

## `UniverseDsmcView` — the device-capturable snapshot

A `Universe` and its state store are host-only and cannot cross onto the device.
`UniverseDsmcView::make(universe)` gathers, once on the host, the raw device
pointers of the DSMC fields plus the two grid scalars into a plain
trivially-copyable struct a device lambda captures by value. Pointer const-ness
mirrors the solver's access — read-only inputs are `const`, written-back counters
are mutable.

```cpp
struct UniverseDsmcView {
    const float* number_particle {};     // read-only per-cell count N
    float*       max_relative_speed {};  // written each step
    float*       max_sigma_g {};         // read then raised
    int*         collision_count {};     // scheduling slot
    const int*   allocated_solver {};    // read-only; null = all solvers own every cell
    int          cell_count {};
    float        cell_volume {};         // collision-rate denominator
    static UniverseDsmcView make(Universe&);
    bool is_complete() const noexcept;
};
```

Each pointer is filled only if the matching field is registered, and stays null
otherwise. Build it directly or through the generic
`universe.view<UniverseDsmcView>()`.

### The `is_complete()` contract

`is_complete()` returns true only when `number_particle`, `max_relative_speed`,
`max_sigma_g`, and `collision_count` are **all** non-null.
**`allocated_solver` is intentionally excluded**: a null owning-solver buffer is
the valid "all solvers own every cell" case, not a missing field. A dedicated test
pins this — `UniverseDsmcView.AllocatedSolverExcludedFromCompleteness` adds *only*
`UniverseAllocatedSolverState` and still expects `is_complete()` to be false
(`tests/atlas/universe/universe_view_tests.cpp`).

The DSMC solver checks completeness before dereferencing: if either this view or
the [fluid view](../fluid/fluid.md#the-is_complete-contract) is incomplete,
`DsmcSolver::solve` (`src/atlas/solver/dsmc/dsmc_solver.cu`) logs a warning and
**performs no collisions** for that call. See [solver](../solver/solver.md).

## Ownership and lifetime

`Universe` is **move-only**: it owns device buffers (through the state store)
whose copy is host-only, so copy operations are deleted and it is passed around as
a `UniverseHostPtr` (`host_unique_ptr<Universe>`). Note `Universe` has no default
constructor — the domain corners and cell size must always be supplied.

A view **owns nothing** — its pointers alias the field buffers inside the
`UniverseState` leaves. They stay valid only until a field is removed, replaced,
or reallocated (e.g. `System` resizing a field in `ensure_universe_state` when the
cell count changed), so a gathered view must be re-gathered after any such
operation.

## See also

- [fluid](../fluid/fluid.md) — the per-particle columns the same solvers read
  alongside these grid fields.
- [buffer](../buffer/buffer.md) — the `DeviceBuffer<T>` owning container every
  field is built on.
- [solver](../solver/solver.md) — the DSMC solver that consumes `UniverseDsmcView`.
