# Fluid

The fluid module is the engine's **Lagrangian particle population**: a
fixed-capacity, structure-of-arrays gas held in device memory. A `Fluid` owns the
per-particle attribute *columns* (position, velocity, species, optional energy
modes) plus the bookkeeping buffers used to drop dead particles. It is one half of
the data model — the moving particles — with the fixed background grid living in
the [universe module](../universe/universe.md). No solver touches a `Fluid`
directly on the device; kernels reach its columns through a trivially-copyable
*view* of raw device pointers (`FluidDsmcView`).

## Files

| File | Role |
|---|---|
| `include/atlas/fluid/fluid.h` | `Fluid` (the particle population) + `Fluid::Builder`; `FluidStateStore = TypeStore<FluidState>`; `FluidHostPtr` |
| `include/atlas/fluid/fluid_state.h` | `FluidState` base + the seven column leaves (position/velocity/species + four optional) |
| `include/atlas/fluid/fluid_view.h` | `FluidDsmcView` — device-capturable snapshot for the DSMC collision kernel |
| `src/atlas/fluid/fluid.cu` | `Fluid` / `Builder` out-of-line definitions, including `compact()` |
| `src/atlas/fluid/fluid_state.cu` | The leaf constructors and `size` / `compact` / `reset` overrides |

## The columns (`FluidState` leaves)

Each attribute is a separate device column — a structure-of-arrays layout so a
kernel touches only the fields it needs and a new attribute can be added without
disturbing the others. Every column is a `FluidState` leaf owning exactly one
`DeviceBuffer`, sized to the fluid's **capacity** (`buffer_size`), not its live
count. The base fixes the shared contract: `size()`, `compact(compact_indices,
kept)` (gather survivors to the front), and `reset()` (zero-fill).

| Leaf | Element | Allocated | Role |
|---|---|---|---|
| `FluidPositionState` | `Float3` (m) | always | Particle positions; read by motion integration, sink despawn tests, the spatial searcher |
| `FluidVelocityState` | `Float3` (m/s) | always | Velocities; read by integration and by the DSMC kernel, which writes post-collision velocities back |
| `FluidSpeciesState` | `std::size_t` | always | Per-particle material-dictionary index; selects mass and cross-section |
| `FluidTemperatureState` | `float` (K) | on demand | Per-particle temperature |
| `FluidTranslationalEnergyState` | `float` (J) | on demand | Translational internal-energy mode |
| `FluidRotationalEnergyState` | `float` (J) | on demand | Rotational internal-energy mode |
| `FluidVibrationalEnergyState` | `float` (J) | on demand | Vibrational internal-energy mode |

Construction (`Fluid(buffer_size)`) seeds exactly the three mandatory columns
(the `.cu` reserves three slots and emplaces position, velocity, species).
Optional columns are registered later via `emplace_state<T>()` / `set_state<T>()`.

## State storage — the `TypeStore`

The columns are held type-erased in a `FluidStateStore` = `TypeStore<FluidState>`
(see `include/atlas/container/type_store.h`), a host-side map keyed on the
concrete C++ leaf type. There is **at most one column per type**, and inserting a
second one of the same type replaces the first. The `Fluid` methods are thin
forwarders over the store:

```cpp
Fluid fluid(1024);                                   // seeds position/velocity/species
fluid.emplace_state<FluidTemperatureState>(1024);    // add an optional column
FluidVelocityState* v = fluid.state<FluidVelocityState>();  // lookup, or nullptr if absent
bool has = fluid.has_state<FluidTemperatureState>();
auto owned = fluid.remove_state<FluidTemperatureState>();   // detach ownership
```

`state<T>()` returns a borrowed pointer (or `nullptr` when the column is not
registered), and is exactly what a view's `make()` uses to decide whether a
pointer can be filled. Iterating the store visits every registered column, which
is how `compact()` gathers all columns with a single index list.

## Capacity vs. live count

- **`buffer_size()`** is the fixed allocation length of every column, chosen at
  construction; it never changes.
- **`particle_count()`** is how many *leading* slots are currently alive; only
  those entries are meaningful. `set_particle_count()` throws `std::out_of_range`
  if asked to exceed the capacity.

Because the capacity is fixed, emission is capped: `System::emit`
(`src/atlas/system/system.cu`) appends spawns starting at the live count and
**stops once `count >= buffer_size`**, so excess spawns for the step are silently
dropped rather than growing the buffer.

## `statistical_weight` and materials

`statistical_weight()` is the DSMC macroscopic scale factor — real molecules
represented per simulated particle — always positive (the builder rejects a
non-positive value with `std::runtime_error`). It defaults to `1.0f`. The DSMC
collision view carries it through to the kernel.

A `Fluid` also carries a `MaterialDictionaryHostPtr` (`materials()`, may be null).
The species column indexes into that dictionary for per-particle mass, collision
cross-section, and energy-mode properties. See
[material](../material/material.md).

## `compact()` — dropping dead particles

`active()` is a per-particle survivor-flag buffer (`DeviceBuffer<int>`, `0`/`1`,
sized to capacity). A removal pass (e.g. `System::mark_survivors`) writes it, then
`compact()`:

1. exclusive-scans the first `particle_count` flags to assign each survivor a
   compacted slot,
2. reads back only the survivor total (the one host/device sync),
3. builds the survivor index list and asks **every** registered column to
   `compact()` itself by that same list, keeping all columns row-aligned,
4. refills the leading flags with `1` and updates the live count.

It is a no-op returning the current count when nothing is alive, and short-circuits
(no gather) when every particle survives.

## `FluidDsmcView` — the device-capturable snapshot

A `Fluid` is a host-only owner of device buffers and cannot be captured into a
device lambda. `FluidDsmcView` is the bridge: `make(fluid)` gathers, once on the
host, the raw device pointers the collision step needs into a plain
trivially-copyable struct that a `__host__ __device__` lambda captures by value.

```cpp
struct FluidDsmcView {
    Float3*            velocity {};            // written by the collision step
    const std::size_t* species {};            // read-only material indices
    int                particle_count {};     // snapshot at gather time
    float              statistical_weight {};
    static FluidDsmcView make(Fluid&);
    bool is_complete() const noexcept;        // velocity && species non-null
};
```

Each column pointer is filled only if that column is registered; a missing column
leaves the pointer null. A trivial-copyability test and the null-pointer defaults
are pinned by `tests/atlas/fluid/fluid_view_tests.cpp`. Build it either directly
(`FluidDsmcView::make(fluid)`) or through the generic `fluid.view<FluidDsmcView>()`.

### The `is_complete()` contract

`is_complete()` returns true only when **both** the `velocity` and `species`
pointers are non-null — i.e. both columns were present. The DSMC solver checks
this before dereferencing: if either the fluid view or the
[universe view](../universe/universe.md#the-is_complete-contract) is incomplete,
`DsmcSolver::solve` (`src/atlas/solver/dsmc/dsmc_solver.cu`) logs a warning and
**performs no collisions** for that call. See [solver](../solver/solver.md).

## Ownership and lifetime

`Fluid` is **move-only**: it owns device buffers whose copy is host-only, so copy
construction/assignment are deleted and it is passed around as a `FluidHostPtr`
(`host_unique_ptr<Fluid>`). The leaves are move-only for the same reason.

A view **owns nothing** — its pointers alias the fluid's buffers. They are valid
only while that fluid is alive and its columns are not reallocated. Any operation
that may move a buffer — `compact()` (which resizes/rebuilds column storage),
removing or replacing a column, or destroying the fluid — invalidates a
previously gathered view, so re-gather it afterwards.

## Construction

Use the fluent `Fluid::Builder` (`with_buffer_size`, `with_particle_count`,
`with_statistical_weight`, `with_materials`), whose `build()` / `make_host_unique()`
validate the parameters (positive weight, initial count within capacity) and then
let the `Fluid` constructor create the three mandatory columns.

```cpp
FluidHostPtr fluid = Fluid::builder()
    .with_buffer_size(1 << 20)
    .with_particle_count(0)
    .with_statistical_weight(1.0e12f)
    .with_materials(dictionary)
    .make_host_unique();
```

## See also

- [universe](../universe/universe.md) — the per-cell grid fields the same solvers
  read alongside the fluid columns.
- [buffer](../buffer/buffer.md) — the `DeviceBuffer<T>` / `HostBuffer<T>` owning
  containers every column is built on.
- [solver](../solver/solver.md) — the DSMC solver that consumes `FluidDsmcView`.
