# Interactive execution and rendering

`src/interactive/` provides a native C++ application around
[Atlas Core](../architecture/overview.md). It accepts JSON configuration and
newline-delimited JSON (JSONL) commands, owns multiple simulation sessions,
records statistics and optional CSV, and can present one session in a native
OpenGL window. It does not use Python or send per-particle state through JSON.

## Components and ownership

```text
stdin JSONL → JsonLinesTransport → JsonCodec → InteractiveApplication
                                         ├─ Server → Session → CoreFactory → Core builders → System
                                         └─ RenderManager → Renderer → WindowTarget
                                                              ↑
                selected Session::scene_view() → RawStateProvider → StateBridge
                                                   └─ ParticleLayer + GeometryLayer
```

`SimulationConfig` is the typed, transport-independent simulation description.
`JsonCodec` decodes it and converts protocol requests and responses.
`JsonLinesTransport` reads one request per line and writes one response per line.
`InteractiveApplication` sends simulation commands to `Server` and rendering
commands to an optional `RenderManager`. `Server` allocates IDs and owns
`Session` objects. Each Session owns one live Core `System`, its original
configuration, run state, statistics, and optional `CsvWriter`. `CoreFactory`
(the compatibility alias is `SystemFactory`) translates configuration to Core
builder calls, including mesh ownership. It does not keep a second set of
physical validation rules.

`RenderManager` selects one Server-owned session and owns `WindowTarget` and
`Renderer`. The renderer owns a `RawStateProvider`, camera, `GeometryLayer`, and
`ParticleLayer`. It receives short-lived read-only scene views, never a mutable
`System`. The session owns no OpenGL resources, and `Renderer` never advances
simulation. `InteractiveApplication::update()` advances running sessions through
Server and then asks the manager to draw when a frame is due. The two lifetimes
are independent.

## Building and starting

The [build guide](../contributing/build.md) covers TBB, CUDA, presets, and
graphics dependencies. `atlas::interactive` is the graphics-free execution
library. `ATLAS_INTERACTIVE_RENDERING=ON` adds `atlas::interactive-rendering` and
links it into the `atlas-interactive` executable; `OFF` leaves the executable
headless. The maintained [Interactive example](../../examples/README.md#native-interactive-window)
uses the same application and renderer composition.

Start the JSONL server with a case:

```bash
./build/interactive-headless-tbb/src/interactive/atlas-interactive \
    --config examples/interactive/cases/cylinder.json
```

This emits a `create` response with `request_id: "startup"` and a `session_id`.
Starting without `--config` lets clients create sessions by command. Commands
arrive on stdin; responses use stdout. Core logs go to stderr. A rendering-enabled
executable can also accept `render_open` and `render_close`; the window requires
a graphics display. The JSONL transport itself remains stdin/stdout.

## Simulation JSON

A complete simulation object requires `dt`, `fluid`, and `universe`. Arrays of
`solvers`, `emitters`, `colliders`, and `sinks`, plus `codec`, are optional. JSON
uses numeric scalars, three-component vectors, and scalar-first quaternions
`[w,x,y,z]`. The [cylinder case](../../examples/interactive/cases/cylinder.json)
is runnable strict JSON; the [template](../../examples/interactive/template.jsonc)
shows the broader field set with comments and is not directly parseable JSON.

| Object | Fields | Meaning |
|---|---|---|
| `fluid` | `buffer_size` (required), `particle_count`, `statistical_weight`, `materials`, `position`, `velocity`, `species`, `temperature`, `translational_energy`, `rotational_energy`, `vibrational_energy` | Fixed particle capacity, live prefix, species table, and optional initial columns. Each supplied initial column has one value per live particle. |
| `materials[]` | `type`, `mass`, `translational_energy`, `rotational_energy`, `vibrational_energy`, `reference_diameter`, `reference_temperature`, `viscosity_index`, `scattering_parameter` | Material types: `molecule`, `atom`, `ion`, `neutron`, `solid`. Solid construction uses its mass. Array order determines species indices. |
| `universe` | `cell_size` (required), `lower_corner` and `upper_corner`, or `geometry`; optional `temperature`, `bulk_velocity`, `field_force`, `gravity`, `thermal_energy`, `knudsen_number` | Domain and fixed grid. A geometry derives the bounds when supplied. Each supplied cell column must match the derived cell count. |
| `solvers[]` | `kernel`, `majorant_sample_pairs`, `majorant_exhaustive_limit` | DSMC kernel: `hard_sphere`, `variable_hard_sphere`, or `variable_soft_sphere`. Solvers run in array order. |
| `emitters[]` | `source`, `generator` | The source places particles and its paired generator initializes their state. |
| `source` | `type`, `unit`, `tolerance`, `spacing` | `surface` or `volume` emission boundary. |
| `generator` | `type`, `species_ratios`, `species_numbers`, `temperature`, `bulk_velocity`, `seed`; distribution fields | `uniform` uses `min_value`/`max_value`; `jittering` uses `base_value`/`jitter_radius`; `maxwell_sigma` uses `sigma`; `maxwell_boltzmann` can use `species_mass` or the Fluid material dictionary. |
| `colliders[]` | `unit`, `momentum_accommodation_coefficient`, `restitution`, `diffuse_sampling` | Isothermal wall; diffuse sampling is `uniform` or `cosine_weighted`. |
| `sinks[]` | `type`, `unit`, `tolerance` | `surface`, `volume`, or `tracing` removal boundary. |
| `codec` | `representative_characteristic_length`, `representative_collision_cross_sectional_area`, `representative_statistical_weight`, `representative_cell_volume` | Optional Knudsen-based per-cell solver selector. |

A `unit` contains a required `geometry` and optional `translation`,
`orientation`, `velocity`, `acceleration`, `angular_velocity`, and
`angular_acceleration`. It combines local geometry with a world pose and
optional motion. Geometry types and their principal fields are:

| `type` | Fields |
|---|---|
| `box` | `lower_corner`, `upper_corner` |
| `circle` | `center`, `normal`, `radius` |
| `cylinder` | `center`, `radius`, `height`, optional `open` |
| `plane` | `normal`, `offset` |
| `sphere` | `center`, `radius` |
| `square` | `center`, `normal`, `side_length` |
| `triangle` | `a`, `b`, `c` |
| `triangle_mesh` | `triangles`, each a three-vertex array |
| `polygonal_prism` | `center`, `side_count`, `radius`, `height` |

`JsonCodec` checks JSON syntax, types, vector dimensions, enum spellings, and
required fields for partial `validate` targets. Core builders perform Atlas
semantic and numerical checks, including particle and cell column lengths and
geometry or policy constraints. The same Core construction path is used for a
complete `validate` and `create`. A decoding or builder exception becomes a
failed response with `success: false` and an `error` string; no partial session
is installed.

## JSONL commands

Requests have `command`, optional caller supplied `request_id`, and a `payload`
object when arguments are needed. The decoder also accepts arguments at the top
level for local tools. `create`, `validate`, and `shutdown` are process scoped;
every other command requires `session_id`. Responses echo `request_id` and
include `success`, plus `message`, `error`, `session_id`, or `status` when relevant.

| Command | Payload | Effect |
|---|---|---|
| `create` | `config`, optional `output` | Build a Session and return its new ID and ready status. |
| `validate` | `target`, `config` | Build a Core object and discard it; no session created. |
| `start` | none | Set a session to running; the application loop advances it. |
| `pause` | none | Stop automatic updates. |
| `step` | optional `step_count` (default 1) | Advance the selected session immediately. |
| `status` | none | Return state, step, simulation time, particle count, source count, sink count. |
| `save` | `path` | Flush CSV and save a Core snapshot below the requested directory. |
| `restart` | none | Rebuild from the original config, reset statistics, and enter ready state. |
| `close` | none | Remove a session; close its window first if selected. |
| `render_open` | none | Open a window for this live session in a rendering build. |
| `render_close` | none | Close the selected session's window, leaving the session alive. |
| `shutdown` | none | Close rendering and request process shutdown. |

`validate` targets are `material`, `geometry`, `unit`, `fluid`, `universe`,
`solver`, `source`, `generator`, `emitter`, `collider`, `sink`, `codec`, and
`simulation`. For `generator`, `config` wraps `generator` and optional
`materials`; `emitter` wraps `source`, `generator`, and optional `materials`. A
complete `simulation` target takes the ordinary simulation object.

```json
{"request_id":"check","command":"validate","payload":{"target":"universe","config":{"lower_corner":[-1,-1,-1],"upper_corner":[1,1,1],"cell_size":0.25}}}
{"request_id":"new","command":"create","payload":{"config":{"dt":0.01,"fluid":{"buffer_size":64},"universe":{"lower_corner":[-1,-1,-1],"upper_corner":[1,1,1],"cell_size":0.25}}}}
{"request_id":"run","session_id":1,"command":"start"}
{"request_id":"view","session_id":1,"command":"render_open"}
{"request_id":"inspect","session_id":1,"command":"status"}
{"request_id":"hide","session_id":1,"command":"render_close"}
{"request_id":"pause","session_id":1,"command":"pause"}
{"request_id":"save","session_id":1,"command":"save","payload":{"path":"snapshots"}}
{"request_id":"reset","session_id":1,"command":"restart"}
{"request_id":"step","session_id":1,"command":"step","payload":{"step_count":3}}
{"request_id":"close","session_id":1,"command":"close"}
{"request_id":"stop","command":"shutdown"}
```

A successful validation response is
`{"request_id":"check","success":true,"message":"valid"}`. A status response
includes a `status` object such as
`{"state":"paused","step":12,"simulation_time":0.12,"particle_count":3,"source_count":0,"sink_count":0}`.
Malformed input returns a failed response and the transport continues reading.
Correlate replies by `request_id`.

The `create` payload may include an output policy:

```json
{"csv_enabled":true,"output_directory":"results","csv_filename":"statistics.csv"}
```

CSV rows are appended after completed steps. Session statistics keep current
and cumulative source and sink counts. `restart` starts a fresh CSV run. `save`
flushes CSV and writes `time_step_<step>/fluid.bin` and `universe.bin` through
Core serialization.

## Rendering and lifetime

`RenderManager` selects one session by ID. Opening the same session again is
idempotent; opening another while a window is active fails until the first is
closed. Each frame obtains a fresh `Session::scene_view()`: live particle buffer
views, current boundary poses and geometry descriptions, and Universe bounds.
These views are borrowed from the live System and stored configuration. They
must not survive a resize, restart, or session destruction. `RawStateProvider`
uploads particle columns to reusable OpenGL buffers, passes geometry poses to
`GeometryLayer`, and clears borrowed geometry references after the frame.
`ParticleLayer` draws particles. Infinite planes are displayed as bounded
patches based on the Universe.

For TBB, `StateBridge` uploads host-accessible Core storage to OpenGL. For CUDA
it normally keeps OpenGL buffers registered with CUDA and copies device to
device. Buffer growth recreates a registration; if registration is unavailable,
it copies through host staging. No JSON or NumPy array is involved.

Rendering never advances the simulation. Closing the native window stops
rendering only; its Server-owned session remains alive and may continue running
or be stepped without a window. `render_close` has the same separation. `close`
and `shutdown` release graphics resources before destroying the selected
session or server state.

A headless build (`ATLAS_INTERACTIVE_RENDERING=OFF`) still accepts all simulation
and validation commands. `render_open` and `render_close` produce a failed
response with `Rendering is unavailable in this build.` Direct calls to
`Server` also reject rendering commands; use `InteractiveApplication` to
coordinate them.

The current transport is JSONL on stdin/stdout. `OffscreenTarget` has no headless
OpenGL context and `FrameStream` has no concrete encoder or network transport,
so the implemented presentation path is the native window. Material leaves are
direct Core value types: their construction does not currently reject every
nonphysical scalar property. The material dictionary builder checks that a
supplied dictionary is nonempty; `validate material` is not a complete physical
material audit.

## Standard development environment

Run this guide's native configure/build commands inside
`python3 scripts/dev.py tbb` or `python3 scripts/dev.py cuda`, after building the
corresponding development image. JSON and rendering development packages are
preinstalled. A headless build still searches for no graphics packages. Window
execution uses `ATLAS_DOCKER_DISPLAY=1` to forward the host X11 display; CUDA
execution additionally uses the host GPU exposed by the launcher. See
[Docker operations](../operations/docker.md#standard-development-environment).
