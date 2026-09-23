# Interactive Execution and Rendering

`src/interactive/` is the native C++ execution, control, and rendering layer for
Atlas. It consumes the core without changing the core dependency direction:

```text
Python binding ───────→ Atlas Core ←────── Interactive
```

The installed `atlas-interactive` executable is a newline-delimited JSON
(JSONL) simulation server. The optional OpenGL renderer is a separate library,
and the native window is an example under `examples/interactive/`. Neither path
uses Python, NumPy, or per-particle JSON transfer.

## Architecture

```text
JSON config / JSON command
        ↓
Transport
        ↓
Server
        ↓
Session
        ↓
SystemFactory
        ↓
Atlas Core
```

| Component | Responsibility |
|---|---|
| `JsonCodec` | Convert JSON documents to configuration and protocol values, and responses back to JSON. |
| `JsonLinesTransport` | Read one request per stdin line and write one response per stdout line. |
| `Server` | Own sessions, allocate session IDs, route commands, remove sessions, and coordinate process shutdown. |
| `Session` | Own one live `System`, step it, collect statistics, save snapshots, restart, and expose read-only render views. |
| `SystemFactory` | Convert an immutable `SimulationConfig` into Atlas builders and native objects. |
| `OutputConfig` | Configure optional CSV output independently of simulation physics. |
| `StateProvider` | Convert a read-only scene view into graphics-side `RenderState`. |
| `StateBridge` | Transfer selected core memory into reusable OpenGL buffers. |
| `Renderer` | Orchestrate the provider, camera, layers, and target without stepping a simulation. |
| `GeometryLayer` | Build and draw current source, collider, and sink geometry at each Unit's current pose. |
| `RenderTarget` | Own where OpenGL output is presented. |

`Session` is the only interactive class that accesses its live `atlas::System`.
It does not expose a public mutable system reference. JSON parsing stays in the
transport layer, while core construction stays in `SystemFactory`.

## Building

CMake options, target names, and backend constraints are maintained in the
[build guide](../contributing/build.md). `atlas::interactive` is the headless
execution library; `atlas::interactive-rendering` adds OpenGL dependencies.

## Starting the JSON server

Start with a simulation file:

```bash
./build/interactive-headless-tbb/src/interactive/atlas-interactive \
    --config examples/interactive/cases/cylinder.json
```

The first stdout line is a `create` response with `request_id` set to
`"startup"` and the allocated `session_id`. Commands then arrive on stdin as
one JSON object per line. Responses are emitted on stdout in the same format.
Atlas log messages are directed to stderr so stdout remains a valid JSONL
stream.

A process may also start without `--config`. Create each session explicitly:

```json
{"request_id":"create-1","command":"create","payload":{"config":{"dt":0.01,"fluid":{"buffer_size":64},"universe":{"lower_corner":[-1,-1,-1],"upper_corner":[1,1,1],"cell_size":0.25}},"output":{"csv_enabled":false}}}
```

`Server` owns every successfully created session until `close` removes it or
the process exits. Invalid configuration returns `success: false` and an error;
it does not install a partial session.

## Simulation configuration

The top-level simulation object has these fields:

| Field | Content |
|---|---|
| `dt` | Positive core timestep. |
| `fluid` | Capacity, live count, statistical weight, materials, and optional initial particle states. |
| `universe` | Bounds or a geometry, cell size, and optional user-facing cell states. |
| `solvers` | DSMC kernel and majorant settings. |
| `emitters` | Source and generator pairs. |
| `colliders` | Isothermal collider Units and wall parameters. |
| `sinks` | Surface, volume, or tracing sink Units. |
| `codec` | Optional Knudsen codec representative values. |

Fluid state arrays use `position`, `velocity`, `species`, `temperature`,
`translational_energy`, `rotational_energy`, and `vibrational_energy`. Initial
particle arrays, when provided, must match `particle_count`. Universe arrays
must match the calculated cell count. Solver-owned fields such as collision counters,
`max_sigma_g`, and allocated-solver state are intentionally absent from normal
configuration.

Geometry `type` accepts `box`, `circle`, `cylinder`, `plane`, `sphere`,
`square`, `triangle`, `triangle_mesh`, and `polygonal_prism`. A Unit combines a
geometry with optional `translation`, quaternion `orientation` in `[w,x,y,z]`
order, `velocity`, `acceleration`, `angular_velocity`, and
`angular_acceleration`.

Generator `type` accepts `uniform`, `jittering`, `maxwell_sigma`, and
`maxwell_boltzmann`. Shared generator fields are `species_ratios`,
`species_numbers`, `temperature`, `bulk_velocity`, and `seed`. Generator-specific
fields are `min_value`, `max_value`, `base_value`, `jitter_radius`, `sigma`, and
`species_mass`. A Maxwell-Boltzmann generator may resolve masses from the Fluid
material dictionary or from explicit `species_mass`.

Material `type` accepts `molecule`, `atom`, `ion`, `neutron`, and `solid`.
Gas-like materials accept mass, energy, reference diameter/temperature,
viscosity index, and scattering parameter. A solid currently uses mass only.
See [`examples/interactive/cases/cylinder.json`](../../examples/interactive/cases/cylinder.json)
for a runnable document.

## JSONL commands

Every request may carry a caller-selected `request_id`. Commands other than
`create` and `shutdown` require `session_id`.

| Command | Payload | Behavior |
|---|---|---|
| `create` | `config`, optional `output` | Validate configuration, build a Session, and return its ID. |
| `start` | none | Enter `running`; the server loop advances the session continuously. |
| `pause` | none | Enter `paused`; automatic updates stop. |
| `step` | optional `step_count` | Run the requested number of steps immediately; default is one. |
| `status` | none | Return state, step, simulation time, particle count, source count, and sink count. |
| `save` | `path` | Flush CSV and write the core snapshot under the requested directory. |
| `restart` | none | Rebuild from the stored original `SimulationConfig`, reset statistics, and enter `ready`. |
| `close` | none | Remove and destroy the selected Session. |
| `shutdown` | none | Stop the server process. |

Examples:

```json
{"request_id":"start-1","session_id":1,"command":"start"}
{"request_id":"pause-1","session_id":1,"command":"pause"}
{"request_id":"step-1","session_id":1,"command":"step","payload":{"step_count":3}}
{"request_id":"status-1","session_id":1,"command":"status"}
{"request_id":"save-1","session_id":1,"command":"save","payload":{"path":"snapshots"}}
{"request_id":"restart-1","session_id":1,"command":"restart"}
{"request_id":"close-1","session_id":1,"command":"close"}
{"request_id":"shutdown-1","command":"shutdown"}
```

A successful status response looks like:

```json
{"request_id":"status-1","session_id":1,"success":true,"message":"status","status":{"state":"paused","step":12,"simulation_time":0.12,"particle_count":3,"source_count":0,"sink_count":0}}
```

Malformed JSON produces a failed response and the transport continues reading
later lines. Application errors likewise produce `success: false` with an
`error` string. Clients should correlate responses through `request_id` rather
than relying only on ordering.

## Output, statistics, and snapshots

The `output` object accepted by `create` is independent of the physical
configuration:

```json
{
  "csv_enabled": true,
  "output_directory": "results",
  "csv_filename": "statistics.csv"
}
```

`Session` converts the core's last-step source and sink counters into a
`SimulationSample`. `SimulationStatistics` keeps the current and cumulative
values for the active run. If CSV is enabled, `CsvWriter` appends a row after
each completed step without retaining all rows in memory.

`restart` resets statistics and reopens the configured CSV as a fresh run.
`save` flushes the CSV before `System::save()` writes
`time_step_<step>/fluid.bin` and `universe.bin`.

The corresponding native C++ construction is:

```cpp
atlas::interactive::SimulationConfig simulation;
simulation.dt = 1.0e-5f;
simulation.fluid.buffer_size = 1024;
simulation.universe.lower_corner = { -1.0f, -1.0f, -1.0f };
simulation.universe.upper_corner = { 1.0f, 1.0f, 1.0f };
simulation.universe.cell_size = 0.1f;

atlas::interactive::OutputConfig output;
output.csv_enabled = true;
output.output_directory = "results";

atlas::interactive::Session session(std::move(simulation), std::move(output));
session.start();
session.update();
session.pause();
```

## Native rendering

The local window example composes rendering independently from JSON control:

```text
Session::update()
    ↓
Session::scene_view()
    ↓
RawStateProvider
    ↓
StateBridge
    ↓
RenderState
    ↓
ParticleLayer + GeometryLayer
    ↓
Renderer
    ↓
WindowTarget
```

`SimulationSceneView` contains a short-lived particle-buffer view, the current
source/collider/sink geometry poses, and the simulation domain. Rendering never
receives `System`. Obtain a new scene view after each simulation step and do not
retain it across operations that may resize, compact, restart, or destroy the
simulation.

`GeometryLayer` explicitly dispatches all nine geometry types. Infinite planes
are shown as finite patches derived from the Universe bounds. Unit motion is
read from the current core pose snapshot, so rendering does not integrate or
duplicate motion.

Run the example, optionally closing after a fixed number of frames:

```bash
./build/tbb-application-release/examples/interactive/atlas-interactive-example
./build/tbb-application-release/examples/interactive/atlas-interactive-example cylinder 10
```

For TBB, `StateBridge` uploads directly from the core CPU buffer into OpenGL.
For CUDA, it persistently registers reusable OpenGL buffers and normally copies
device to device through CUDA/OpenGL interop. Registration is recreated only
when a graphics buffer grows. A host-staged fallback is used only when the
active OpenGL device cannot interoperate with the CUDA device.

## Current limits

`OffscreenTarget` and `FrameStream` remain future boundaries. There is no
headless OpenGL context, frame encoder, TCP/WebSocket transport, or camera wire
protocol yet. The implemented control transport is JSONL over stdin/stdout.

Future external clients will receive rendered frames through `FrameStream`.
They will not receive raw Fluid arrays through Python lists, NumPy, JSON, or a
frontend-owned upload path.
