# Serialization

A protobuf-backed snapshot codec that captures a live `Fluid` or `Universe` to a
binary file and rebuilds it later. It is not part of the `System::update()` step
pipeline; it is a cross-cutting service the runtime calls when it wants to persist
state — `System::save(directory)` writes one `fluid.bin` and one `universe.bin`
per checkpoint. The design bridges three representations: the engine's
host/device buffers, the generated protobuf messages (`atlas_snapshot.pb.h`), and
the on-disk byte stream. Every device column is first pulled to the host, then
blitted verbatim into a `RawBuffer` byte payload, so save/load is a plain
`memcpy` with an element-count check rather than a per-element re-encode.

## Files

| File | Role |
|---|---|
| `include/atlas/serialization/protobuf_snapshot.h` | Public API: the `FluidBinarySnapshot` / `UniverseBinarySnapshot` host structs and the six free functions (`save_*`, `load_*`, `restore_*`) |
| `src/atlas/serialization/protobuf_snapshot.cpp` | The codec: the anonymous-namespace plumbing (pack/unpack, message I/O, material and state encoders) plus the six public entry points |
| `src/atlas/serialization/proto/atlas_snapshot.proto` | The proto3 wire schema, compiled by `protoc` into `atlas_snapshot.pb.{h,cc}` |

## The wire schema (`atlas_snapshot.proto`)

The schema (`package atlas.proto`) defines the on-disk format. Field numbers are
part of that format: they are never renumbered or reused, and new fields are added
only at fresh numbers so old snapshots keep parsing.

- **`Vector3`** — `x`/`y`/`z` stored as `double`, so a snapshot survives
  independently of the engine's runtime float width. The codec narrows back to
  `float` on read (`read_vector3`) and widens on write (`set_vector3`).
- **`RawBuffer`** — a flat, untyped byte blob for one device column:
  `element_count` plus `bytes data`. The bytes are the raw little-endian element
  array; the reader verifies `data.size() == element_count * sizeof(element)`
  before reinterpreting.
- **`Material`** — one `MaterialDictionary` entry: the `MaterialType`
  discriminant (`type`, as `int32`) plus every VHS/VSS property (`mass`,
  translational/rotational/vibrational energy, reference diameter and
  temperature, viscosity index, scattering parameter), each widened to `double`.
  A solid stores only mass; its other fields read back as constants and
  round-trip harmlessly.
- **`ScalarType`** enum — the element precision the buffers were written with.
  The codec only writes and accepts `SCALAR_FLOAT32`; `SCALAR_FLOAT64` is
  reserved for a future double-precision build.
- **`FluidSnapshot`** — a complete capture of one fluid: `version`,
  `scalar_type`, `buffer_size`, `particle_count`, `statistical_weight`, a
  `repeated Material materials` species table, and one `RawBuffer` per column
  (`positions`, `velocities`, `species`, `active`, `temperature`,
  `translational_energy`, `rotational_energy`, `vibrational_energy`). Each energy
  state pairs its buffer with an explicit `includes_*_state` bool, because a
  proto3 message field cannot distinguish "absent" from "present but empty". Tags
  `6, 7` are `reserved` (the former MaterialProperty table and generator
  operators, no longer owned by the fluid).
- **`UniverseStateKind`** enum + **`UniverseState`** message — one per-cell field.
  `UniverseState` bundles three typed `RawBuffer`s (`scalar_buffer`,
  `vector_buffer`, `int_buffer`); `kind` selects which one is meaningful, keeping
  the outer list homogeneous.
- **`UniverseSnapshot`** — grid geometry (`lower_corner`, `upper_corner`,
  `cell_size`) plus a `repeated UniverseState states`. Unlike the fluid, absent
  fields are simply omitted from `states`, so no `includes_*` flags are needed.

## Host-side decoded structs

`load_*` return plain-data structs holding host memory only — no `Fluid`/
`Universe` object and no device allocation:

- **`FluidBinarySnapshot`** — `buffer_size`, `particle_count`,
  `statistical_weight`, a `HostBuffer<Material> materials`, and one
  `std::optional<HostBuffer<…>>` per column. The `std::optional` distinguishes "no
  such state was attached" (`std::nullopt`) from "attached but empty". When
  present, every buffer is exactly `buffer_size` long — `load_fluid_binary`
  enforces that invariant before returning.
- **`UniverseBinarySnapshot`** — the grid geometry (always present) plus one
  `std::optional<HostBuffer<…>>` per per-cell field, filled by dispatching on the
  stored `UniverseStateKind`.

## Save path

`save_fluid_binary(fluid, path)` and `save_universe_binary(universe, path)` stamp
the header (`version = kAtlasSnapshotVersion`, `scalar_type = SCALAR_FLOAT32`),
copy each attached state off the device into a `HostBuffer`, and blit it into a
`RawBuffer`. Both truncate any existing file (`std::ios::trunc`) and throw on an
open or serialize failure.

Both paths keep a `known_state_count` tally of the states they recognize and
compare it against the object's total state count at the end; a mismatch means the
object holds a state type the codec cannot encode, and the function throws rather
than silently drop it on restore. For the fluid, the `active` flag column is
intrinsic (always written) and deliberately excluded from that tally; the four
energy states set their `includes_*_state` flag when written. The universe path
appends each state to the `states` list tagged with its `UniverseStateKind`,
routing the payload into the scalar, vector, or int buffer as appropriate.

## Load and restore paths

`load_fluid_binary(path)` / `load_universe_binary(path)` parse the file and
return the host struct. Both reject a `version` other than
`kAtlasSnapshotVersion` and a `scalar_type` other than `SCALAR_FLOAT32`
(`validate_scalar_type`). The fluid loader infers presence of
positions/velocities/species/active from a nonzero `element_count` (these predate
the flags), reads the energy states via their `includes_*` flags, then checks
that `particle_count <= buffer_size` and that every present column is exactly
`buffer_size` long. The universe loader switches on each `UniverseState.kind` to
route the buffer into the matching optional slot and throws on an unrecognized
kind.

`restore_fluid(path)` / `restore_universe(path)` are convenience wrappers:
`load_*` followed by the object's `Builder`, then uploading each present host
buffer back to device memory as the matching `Fluid*State` / `Universe*State`.
`restore_fluid` rebuilds the `MaterialDictionary` when the snapshot carried one
(leaving it null otherwise) and assigns the `active` column directly (it is a
built-in column, not a pluggable state). Both return an owning host pointer to the
reconstructed, device-resident object.

## How `System` drives it

`System::save(const std::filesystem::path& directory)` (in
`src/atlas/system/system.cu`) writes a per-step checkpoint. The subdirectory name
comes from `System::snapshot_directory_name(step)`, which is simply
`"time_step_" + std::to_string(step)`:

```cpp
const std::filesystem::path step_directory = directory / snapshot_directory_name(_step);
std::filesystem::create_directories(step_directory);
atlas::save_fluid_binary(*_fluid, (step_directory / "fluid.bin").string());
atlas::save_universe_binary(*_universe, (step_directory / "universe.bin").string());
```

So a save of `directory` at step 42 produces
`directory/time_step_42/fluid.bin` and `directory/time_step_42/universe.bin`.

## Protobuf dependency and schema evolution

Protobuf is a vendored in-tree dependency (`external/protobuf`; see
[dependencies.md](../../guidelines/dependencies.md)). The build compiles both the
runtime and `protoc` from source, runs `protoc` on `atlas_snapshot.proto` into a
generated `atlas_snapshot.pb.{h,cc}`, and links them into the `atlas-serialization`
static library alongside `protobuf_snapshot.cpp`. That translation unit stays
`.cpp` (never routed through `nvcc`): protobuf's `message_lite.h` uses a construct
`nvcc`'s frontend rejects, and it holds no device code. `atlas-serialization`
links `protobuf::libprotobuf` and forms a deliberate two-way link cycle with the
`atlas` engine library (the engine's save path calls into serialization, and
serialization reads engine getters/state types back).

The schema is versioned defensively:

- **`version`** (`kAtlasSnapshotVersion = 1`) is written into every message and
  rejected on mismatch — there is no cross-version migration; an old version
  simply fails to load.
- **`scalar_type`** guards element precision: only `SCALAR_FLOAT32` is produced or
  accepted today, with `SCALAR_FLOAT64` reserved for a future build.
- **`Vector3` is double-precision on the wire** so the file is independent of the
  engine's runtime float width.
- **Field numbers are frozen**: `reserved 6, 7` in `FluidSnapshot` blocks reuse of
  retired tags, and the schema comment requires new fields to be added only at
  fresh numbers so old snapshots keep parsing.
- **`includes_*` flags** on the fluid's optional states, and **omission from the
  `states` list** on the universe, both encode "absent vs. present-but-empty" that
  proto3 field presence alone cannot.
