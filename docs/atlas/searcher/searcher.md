# Searcher

The searcher module is the engine's **spatial index**: it buckets fluid
particles into a uniform grid so that a DSMC solver can iterate the particles of
one cell without scanning the whole population. `SpatialHashingSearcher` runs a
standard sort-based spatial hash on the device and hands out a
trivially-copyable `SpatialHashingSearcherView` — four raw device pointers — that
a `__host__ __device__` lambda captures by value to answer neighbour queries on
the GPU.

It is the *search* phase of the step pipeline. `System::search()` calls
`classify()` once per step with the fluid positions and the universe's per-cell
particle-count field; `System::solve()` then gathers the `view()` once and passes
it to every solver. The grid the searcher indexes is the same one the
[`Universe`](../universe/universe.md) defines, so a searcher cell's linear key is
exactly the universe cell index, and the [`DsmcSolver`](../solver/solver.md)
reads `cell_start`/`cell_end`/`sorted_index` directly by cell.

## Files

| File | Role |
|---|---|
| `include/atlas/searcher/spatial_hashing_searcher.h` | `SpatialHashingSearcher` (grid + classify pipeline) and its fluent `Builder` |
| `include/atlas/searcher/spatial_hashing_searcher_view.h` | `SpatialHashingSearcherView` — the trivially-copyable, device-capturable pointer bundle |
| `src/atlas/searcher/spatial_hashing_searcher.cu` | Constructors, the four device passes, `reset`, `view`, and the `Builder` |

## The uniform grid

A searcher is a cubic uniform grid described by five host-side fields:

| Field | Meaning |
|---|---|
| `lower_corner` (`Float3`) | World-space minimum corner the grid is anchored at |
| `cell_size` (`float`) | Edge length of a cubic cell, in world units; must be `> 0` |
| `inverse_cell_size` (`float`) | Cached `1 / cell_size`, so mapping a position to a cell avoids a per-particle divide |
| `grid_size` (`Int3`) | Per-axis cell counts; each component must be `>= 1` |
| `cell_count` (`int`) | Cached `grid_size.x * grid_size.y * grid_size.z`; the length of `cell_start`/`cell_end` |
| `particle_count` (`size_t`) | Current key/index buffer length; zero after `reset()` |

Both `inverse_cell_size` and `cell_count` are derived once in the constructor and
cached; the accessors (`lower_corner()`, `cell_size()`, `inverse_cell_size()`,
`grid_size()`, `cell_count()`, `particle_count()`) are `const noexcept` reads.

The normal way to build one is the fluent `Builder`, usually seeded straight from
the universe so the two grids coincide:

```cpp
SpatialHashingSearcher searcher = SpatialHashingSearcher::builder()
    .with_universe(universe)   // copies lower_corner, cell_size, grid_size
    .build();
```

`with_universe` copies `lower_corner`, `cell_size` and `grid_size` from the
`Universe`; the individual `with_lower_corner` / `with_cell_size` /
`with_grid_size` setters override them. `build()` calls `validate()` first, which
throws `std::invalid_argument` if `cell_size <= 0` or any grid axis is `< 1`, then
constructs the searcher (allocating its per-cell arrays). `make_host_shared()`
does the same and wraps the result in a `host_shared_ptr` — the form
`System` holds (`SpatialHashingSearcherHostPtr`).

### The cell-index convention

Two static helpers, both `ATLAS_ALL_DEVICE` (callable on host and device), define
the mapping between a world position and a cell key. Keeping them static and
public lets the extended device lambdas inside `classify()` — and inside a
caller's neighbour query — invoke them by name; that is also why the per-step
methods are public (nvcc forbids an extended lambda inside a private member).

- **`cell_for(position, lower_corner, inverse_cell_size, grid_size) → Int3`**
  computes `floor((position - lower_corner) * inverse_cell_size)` and then
  **clamps** the result to `[0, grid_size - 1]` on every axis. Two consequences,
  both exercised by the tests:
  - A point exactly on an interior cell boundary lands in the **upper** of the
    two cells, because `floor(1.0) == 1`.
  - A point on or beyond the domain boundary is attributed to the nearest **edge
    cell** rather than producing an out-of-range key — so a neighbour walk that
    reaches past the far corner stays in bounds.
- **`linear_key(ix, iy, iz, grid_size) → uint32_t`** (with an `Int3` overload)
  flattens in-range cell coordinates **row-major** — x fastest, then y, then z:
  `ix + iy*Gx + iz*Gx*Gy`. It does no bounds check; feed it only coordinates
  produced by `cell_for` (or otherwise known in range).
- **`contains_cell(cell, grid_size) → bool`** reports membership **without**
  clamping (`0 <= cell < grid_size` on every axis). Use it when stepping to a
  neighbour cell that may fall outside the domain and you want to skip it rather
  than fold it back onto an edge.

## The classify pipeline

`classify(positions, number_particle, particle_count)` runs the four-step
sort-based spatial hash, each step a single device pass, in exactly this order:

1. **`compute_keys`** — one device thread per particle writes
   `keys[i] = linear_key(cell_for(positions[i], ...))` and seeds
   `indices[i] = i`.
2. **`sort_keys`** — a key-sort (`parallel_sort_by_key<device>`, see
   [`parallel`](../parallel/parallel.md)) over the first `particle_count` entries.
   Afterwards `keys` is ascending and `indices` is permuted to match, so every
   cell's particle indices form one contiguous run.
3. **`build_cell_ranges`** — first refills every `cell_start`/`cell_end` entry
   with the `-1` sentinel, then, with one thread per sorted particle, detects run
   boundaries in the sorted `keys`: a slot whose key differs from its predecessor
   (or `i == 0`) opens its cell's range at `i`; a slot whose key differs from its
   successor (or `i == alive - 1`) closes it at `i + 1` (an **exclusive** end).
   Cells with no particles keep the `-1` sentinel.
4. **`write_cell_counts`** — publishes `count = end - start` (0 for empty cells,
   whose `start` is `-1`) as a `float` into each cell of the
   `UniverseNumberParticleState`. This is what the DSMC solver and Knudsen codec
   read.

The result is two device arrays parallel to the sorted order — `cell_key` (the
sorted linear keys) and `sorted_index` (a **permutation of `[0, particle_count)`**;
the tests confirm the sorted indices are exactly the input indices reordered) —
plus two per-cell arrays, `cell_start` and `cell_end`, indexed by linear cell key.

### The `-1` empty sentinel

A cell that no particle falls into keeps `cell_start == cell_end == -1`. The
sentinel exists to distinguish a genuinely empty cell from the legitimate
half-open range `[0, 0)` of cell 0: a consumer tests `cell_start[c] < 0` to skip
empty cells. This is why `reset()` — and the default constructor — fill both
arrays with `-1` rather than `0`.

### Invalid input

`classify()` rejects unusable input rather than trusting it. If `positions` is
null, `particle_count <= 0`, or `particle_count` is larger than the position
buffer, the searcher is `reset()` to the empty state and only the (all-zero) cell
counts are published, so the resulting `view()` reports no particles instead of
letting a kernel read out of bounds. `number_particle` may itself be null: then
the counts are skipped with a warning (the DSMC solver and the Knudsen codec need
them). If `number_particle`'s length differs from `cell_count`, the counts are
also skipped with a warning.

`reset()` returns the searcher to the "nothing classified" state — key/index
arrays cleared to length 0, every `cell_start`/`cell_end` refilled with `-1` —
while leaving the grid geometry untouched.

## `SpatialHashingSearcherView`

`view()` gathers the four device pointers into a flat, trivially-copyable struct:

```cpp
struct SpatialHashingSearcherView {
    const std::uint32_t* cell_key;      // per sorted slot: linear cell key, ascending
    const int*           sorted_index;  // per sorted slot: original particle index
    const int*           cell_start;    // per cell key: first sorted slot, or -1
    const int*           cell_end;      // per cell key: one-past-last sorted slot, or -1
};
```

`cell_key` and `sorted_index` are indexed by **sorted particle slot**
(`0 .. particle_count-1`); `cell_start` and `cell_end` are indexed by **linear
cell key** (`0 .. cell_count-1`). A cell's particles occupy the half-open range
`sorted_index[cell_start[key] .. cell_end[key])`.

The `SpatialHashingSearcher` owns `DeviceBuffer`s, whose lifetime is host-only and
which a device lambda cannot capture; the view exposes their contents as bare
pointers instead. It neither owns nor frees anything, and its pointers stay valid
only until the originating searcher is re-`classify()`-ed, `reset()`, or
destroyed. A default-constructed (all-null) view is the "has not classified yet"
state; the DSMC solver treats any null pointer as "no spatial data available" and
bails out.

### The neighbour-query idiom

A device lambda captures the view by value and, for a query sphere, walks the
cells overlapping the sphere's AABB — corner to corner via `cell_for` (whose
clamping keeps the walk in bounds) — and, for each cell, scans
`[cell_start[c], cell_end[c])` through `sorted_index`:

```cpp
const SpatialHashingSearcherView view = searcher.view();
// ... inside a [=] ATLAS_ALL_DEVICE(...) lambda:
const Int3 low  = SpatialHashingSearcher::cell_for(query - Float3(radius), lower, inv, grid);
const Int3 high = SpatialHashingSearcher::cell_for(query + Float3(radius), lower, inv, grid);
for (int iz = low.z; iz <= high.z; ++iz)
for (int iy = low.y; iy <= high.y; ++iy)
for (int ix = low.x; ix <= high.x; ++ix) {
    const std::uint32_t key = SpatialHashingSearcher::linear_key(ix, iy, iz, grid);
    const int start = view.cell_start[key];
    if (start < 0) continue;                       // empty cell
    const int end = view.cell_end[key];
    for (int slot = start; slot < end; ++slot) {
        const int particle = view.sorted_index[slot];
        // ... test particle against the query ...
    }
}
```

The DSMC solver uses the same access pattern per cell: `begin = cell_start[cell]`,
`end = cell_end[cell]`, then `sorted_index[begin + local]` to pick collision
partners (`src/atlas/solver/dsmc/dsmc_solver.cu`).

## Ownership and copy/move

`SpatialHashingSearcher` declares **no** special member functions, so it has the
implicitly-generated copy and move constructors and assignments. It is **not**
move-only: the tests move-construct one (`MoveConstructionPreservesGeometryAnd...`)
but nothing deletes copy. In practice the `System` holds it through a
`SpatialHashingSearcherHostPtr` (`host_shared_ptr`) and captures its GPU state
into device kernels only through the pointer-bundle `view()`, never by copying the
object into a kernel.

## Deliberately absent

- **No neighbour-query method on the searcher.** The searcher only *builds* the
  index; the query loop lives at the call site (the test helper, the DSMC
  solver). This mirrors the collider, where the per-particle loop lives outside
  the leaf.
- **No host-side classification path.** Every step is a device pass; there is no
  CPU fallback beyond what the backend split in `parallel`/`scan` already
  provides.
- **`SpatialHashingSearcherDevicePtr`** (`device_shared_ptr` alias) is defined for
  symmetry with other modules but, like the other `*DevicePtr` aliases, has no
  producer — the searcher is always held as a `host_shared_ptr` and reaches the
  device through `view()` (see [`memory`](../memory/memory.md)).

## Not implemented

Nothing in the classify pipeline is stubbed or unreachable — all four passes run
on every `classify()` with usable input, and both per-cell arrays and both sorted
arrays are consumed by the DSMC solver and the tests. The only unused surface is
the `SpatialHashingSearcherDevicePtr` alias noted above (an extension point for a
device-capturable owner that no module has adopted, matching the `memory` module's
`*DevicePtr` story).

## Extending

- **A different index structure** (e.g. a hierarchical grid) would keep the same
  public shape: a host object that owns `DeviceBuffer`s, a `classify()` that fills
  them on the device, and a small trivially-copyable view of raw pointers for
  kernels to capture. Preserve the `-1`-sentinel meaning of `cell_start`/
  `cell_end` if solvers are to consume it unchanged.
- **A new grid parameter** goes through the `Builder`: add a `with_*` setter, a
  field, and a `validate()` check that throws `std::invalid_argument` on a bad
  value — the same pattern `cell_size > 0` and `grid_size >= 1` already follow.
- The classify passes are public only to satisfy nvcc's extended-lambda rule; they
  are not a general API. Callers should use `classify()`, and read results through
  `view()`.

## Recently fixed

The default constructor now calls `reset()`, exactly as the explicit constructor
does, so `cell_start`/`cell_end` are sized to `cell_count()` and filled with the
`-1` sentinel. It was previously `= default`, which left the grid reporting one
cell (`cell_count() == 1`) while backing **no** storage — a consumer that read
`cell_count()` entries from `cell_start()` would have run off the end of an empty
buffer. The `DefaultConstructionBacksTheCellsItReports` test now guards this.
</content>
</invoke>
