# Observer

The `Observer` is the engine's periodic diagnostics writer: every `interval()`
steps it is asked to `observe()` the current [`Fluid`](../fluid/fluid.md) and
[`Universe`](../universe/universe.md) and writes a snapshot of CSV files under
`output_directory()/data`. Between snapshots it holds nothing but two small
device-side counter buffers. It does not own, mutate, or step the simulation, so
a run with no observer behaves identically minus the file output. It is driven by
the [`System`](../system/system.md): `System::update()` calls
`observer->observe(fluid, universe, step)` after each step, and the two counter
matrices are filled from `System::record_spawned` (in `emit`) and
`System::mark_survivors` (in `remove`).

Like the other device-owning types it is **move-only** (its counters are
`DeviceBuffer<int>`, i.e. `thrust::device_vector`, whose copy is host-only and an
alias trap) and is constructed through its `Builder`. The observer is a distinct
output path from `System::save()`, which writes durable protobuf `.bin`
snapshots; the observer writes human-readable CSV on an interval.

## Files

| File | Role |
|---|---|
| `include/atlas/observer/observer.h` | `Observer`, its nested `Builder`, and the `ObserverHostPtr` / `ObserverDevicePtr` aliases |
| `src/atlas/observer/observer.cu` | `observe`, `resize_counters`, `reset_counters`, the CSV helpers, and the `Builder` |

## What it records

Beyond dumping fluid and universe state, the observer keeps two per-entity /
per-species counter matrices, both device-resident so kernels can increment them
directly:

- `spawned()` — a row-major `[source][species]` tally of emitted particles,
  length `source_count * species_count`;
- `despawned()` — a row-major `[sink][species]` tally of removed particles,
  length `sink_count * species_count`.

The observer only **sizes, resets, and serializes** these buffers; the atomic
accumulation happens inside `System`'s device kernels, which raw-pointer into the
buffers exposed by the mutable `spawned()` / `despawned()` accessors.
`species_count()` returns the shared column stride (`0` before the counters are
sized).

### `resize_counters` — row-major sizing

```cpp
void Observer::resize_counters(std::size_t source_count,
                               std::size_t sink_count,
                               std::size_t species_count);
```

Records `species_count` and `assign()`s the two buffers to
`source_count * species_count` and `sink_count * species_count` respectively,
laid out row-major as `[entity][species]`, zero-filling both in the same call.
`System`'s constructor calls it once the source, sink, and species counts are
known (species count comes from the fluid's material dictionary, defaulting to a
single implicit species). Because `assign()` both resizes and clears, no separate
reset is needed straight afterwards.

### `reset_counters`

Zeroes both buffers in place with a device fill, preserving the sizes
`resize_counters` established. Use it to clear the running tallies (for instance
per output window) without disturbing the layout.

## The `observe` interval gate

```cpp
if (_interval == 0 || step % _interval != 0) {
    return;
}
```

`observe()` is guarded up front: a **zero interval disables output entirely**,
and a non-zero interval acts only on steps that are exact multiples of it. Every
other step returns immediately, which matters because a snapshot copies device
buffers to the host to serialize them — a synchronizing, comparatively expensive
operation that the gate is what keeps affordable. `System` always calls
`observe()` after incrementing the step counter; the gate alone decides whether a
step writes files.

## The CSV serialization format

On a snapshot step, `observe()` creates `output_directory()/data` and writes up
to four files, each suffixed with the step number (`_<step>.csv`). A file is
**skipped entirely when its subject is empty**, so the set of files present
records what actually existed.

| File | Index column | One row per | Written when |
|---|---|---|---|
| `fluid_<step>.csv` | `particle` | live particle | `fluid.particle_count() > 0` |
| `universe_<step>.csv` | `cell` | grid cell | `universe.cell_count() > 0` |
| `source_<step>.csv` | `source` | emission source | `spawned()` is non-empty |
| `sink_<step>.csv` | `sink` | removal sink | `despawned()` is non-empty |

Every file starts with a header row: the index-column label followed by one label
per data column, then one data row per index (the index value, then each column's
value). All values are written as `float`.

**State-derived files (fluid, universe).** Columns are assembled by probing the
object for each known state. An absent state simply contributes no column, and a
state whose host copy is shorter than the row count is skipped too, so the header
row names exactly the states the object actually carried — the file is
self-describing. Scalar states become one column; `Float3` vector states expand
into three columns suffixed `_x` / `_y` / `_z`. The fluid file covers position,
velocity, species, the survivor `active` flag (stored on the fluid itself, not as
a tagged state, so it is appended by hand), and the temperature and
translational / rotational / vibrational energy states when present. The universe
file covers temperature, bulk velocity, field force, gravity, max relative speed,
max sigma-g, thermal energy, number-particle, collision count, Knudsen number,
and the allocated-solver column, each emitted only if registered.

**Counter files (source, sink).** These transpose the flat row-major
`[entity][species]` counter into one `species_<k>` column per species and one row
per entity, deriving the entity count as `buffer_length / species_count`. When
`species_count` is `0` the file is skipped, which also avoids a divide-by-zero.
The test `Observer.ObserveSerializesSpawnCountersToCsv` pins the exact shape: a
two-source, three-species spawn matrix `{1,2,3,4,5,6}` yields

```
source,species_0,species_1,species_2
0,1,2,3
1,4,5,6
```

with no `sink_<step>.csv` produced because no sinks were configured (an empty
`despawned()` buffer).

`observe()` throws `std::runtime_error` if an output file cannot be opened for
writing.

## `Observer::Builder`

The builder collects the two build-time settings and produces an `Observer`
(`build()`) or an `ObserverHostPtr` (`make_host_shared()` — the form `System`
holds):

| Setter | Effect | Default |
|---|---|---|
| `with_interval(std::size_t)` | Snapshot cadence in steps; `0` disables output | `0` |
| `with_output_directory(std::filesystem::path)` | Root directory; snapshots go in its `data` child | empty |

There is no `validate()` step: both fields have valid defaults (a zero interval
simply disables output). The counter buffers are **not** configured here — they
are left empty and sized later by `resize_counters()` once the source, sink, and
species counts are known, which is what `System`'s constructor does.

## Aliases

- `ObserverHostPtr` = `host_shared_ptr<Observer>` — the owning handle `System`
  holds (the observer must outlive individual step calls).
- `ObserverDevicePtr` = `device_shared_ptr<Observer>` — provided only for
  symmetry with other modules; unused here.
