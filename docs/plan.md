# Atlas Interactive Refactor / JSON Server / Geometry Rendering

Work on `Wontae-Lee/atlas-engine-dev` from the latest `main` (inspected baseline: `6d880440`). Preserve TBB/CUDA behavior and existing core physics.

## Goal

Refactor `src/interactive` to:

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

Rules:

- `Transport`: JSON I/O/serialization only.
- `Server`: owns, creates, routes, and removes sessions.
- `Session`: owns exactly one live `atlas::System` and manages one simulation.
- `SystemFactory`: converts `SimulationConfig` into core builders/objects.
- Rendering must never access `atlas::System` directly.
- Do not put simulation logic in Transport or Server.

## 1. JSON config and transport

Add `config/`:

```text
config/
├── simulation_config.h
├── output_config.h
└── system_factory.{h,cpp}
```

Replace `SessionConfig` with `OutputConfig`.

Replace the current callback alias `SystemFactory = std::function<...>` with a concrete builder/factory that creates a `System` from `SimulationConfig`.

`SimulationConfig` must represent all currently user-configurable core construction data:

- `dt`
- Fluid: `buffer_size`, `particle_count`, `statistical_weight`, materials, optional initial particle states
- Universe: bounds or geometry, `cell_size`, optional user-facing initial fields
- DSMC solvers: kernel type, majorant parameters
- Emitters: Source + Generator
- Sources: surface/volume, Unit, tolerance, spacing
- Generators: uniform/jittering/maxwell_sigma/maxwell_boltzmann and their current builder parameters
- Unit: geometry, pose, velocity, acceleration, angular velocity, angular acceleration
- Collider: isothermal, MAC, restitution, diffuse sampling
- Sinks: surface/volume/tracing
- Codec: Knudsen representative parameters
- Geometry: all current geometry types
- Materials: molecule/atom/ion/neutron/solid and current material properties

Do not expose solver-owned/internal states such as collision counters, allocated solver, max-sigma-g, etc. as normal config inputs.

Add JSON codec under `transport/`. Use a real JSON library; do not hand-write a parser. Keep JSON conversion separate from `SimulationConfig` and core construction.

Implement a first concrete transport as newline-delimited JSON over stdin/stdout so an external process can spawn `atlas-interactive` and communicate immediately. Keep the transport abstraction replaceable by TCP/WebSocket later.

Support loading the initial simulation from a JSON file, e.g.:

```bash
atlas-interactive --config simulation.json
```

After startup, commands use JSONL stdin and responses use JSONL stdout.

## 2. Protocol and Server

Keep `Command`, `Request`, `Response`, but update them for external sessions.

Required commands:

```text
create
start
pause
step
status
save
restart
close
shutdown
```

Requests/responses must support:

```text
request_id
session_id
command
command-specific payload
success/error
status/session_id payload where applicable
```

`Server` must own sessions, e.g. a map keyed by `SessionId`.

Responsibilities:

- `create`: validate config, create Session, return session id
- simulation commands: route to the selected Session
- `close`: erase that Session
- `shutdown`: server/process-level shutdown

Remove server-level lifecycle from Session:

- remove `Session::_shutdown_requested`
- remove `Session::shutdown_requested()`
- remove Session handling of `shutdown`
- remove `Session::close()` if close is implemented by Server ownership
- remove `SessionState::empty`; a live Session should own a live System
- keep `ready`, `running`, `paused`

`restart` remains a Session operation and rebuilds the System from its stored `SimulationConfig`.

## 3. Session

Session remains the sole interactive gateway to the live core System.

Keep:

- start/pause/step/update
- centralized `advance_once()`
- status
- statistics
- CSV output
- save
- restart
- render/scene views

Do not expose a public mutable `System&`.

Keep `SimulationSample`, `SimulationStatistics`, and `CsvWriter`.

## 4. Main executable

Replace the current hard-coded `src/interactive/main.cpp` simulation/window demo.

`atlas-interactive` becomes the JSON server executable:

```text
load JSON config
→ create Server/session
→ read JSONL requests
→ emit JSONL responses
```

Keep the native local window demo only under `examples/interactive/`.

## 5. Geometry rendering

Implement `GeometryLayer`; it must no longer be a no-op.

Support every current `GeometryType` with an explicit render/mesh path:

```text
box
circle
cylinder
plane
sphere
square
triangle
triangle_mesh
polygonal_prism
```

One `GeometryLayer` with explicit type dispatch is preferred; do not create empty placeholder classes.

Requirements:

- render geometry from current simulation state/pose, including moving Units
- support geometry belonging to colliders, sources, and sinks
- preserve `Session` as the only interactive class accessing System
- add a read-only scene/geometry view from Session as needed
- if core access is missing, add only narrow read-only snapshot/access APIs; do not expose mutable core containers
- do not duplicate/integrate Unit motion independently in rendering
- for CUDA device-resident collider/sink metadata, a small device→host metadata copy for rendering is acceptable
- keep triangle-mesh host/render ownership valid; do not retain dangling device/view pointers
- infinite Plane must use a finite visualization patch derived from the simulation domain or an explicit render extent
- geometry rendering must work with the existing camera/renderer/target model
- keep particle rendering unchanged

Add suitable view types, for example:

```text
view/
├── simulation_buffer_view.h
├── simulation_render_view.h
├── geometry_render_view.h
└── simulation_scene_view.h
```

Exact names may vary if a cleaner structure is found.

## 6. Remove dead placeholders

Delete:

```text
rendering/state/interpolation_state_provider.*
```

Remove it from CMake/docs/tests.

`OffscreenTarget` may remain because its future role is clear, but keep its unimplemented status explicit.

Move `transport/frame_stream.*` out of command transport if useful, preferably:

```text
rendering/stream/frame_stream.*
```

It is rendered-frame streaming, not JSON control transport.

## 7. Output config

Rename:

```text
SessionConfig → OutputConfig
```

Keep:

```text
csv_enabled
output_directory
csv_filename
```

Do not mix simulation physics configuration with output configuration.

## 8. Tests

Add/update tests for:

- JSON config parsing and validation
- representative config covering all major config branches
- `SystemFactory` builds a valid System
- Server create/route/close/shutdown
- Session start/pause/step/restart/status/statistics/save
- restart rebuilds from the original config
- invalid session id / malformed JSON / invalid config errors
- GeometryLayer dispatch covers all 9 geometry types
- moving Unit geometry uses current pose
- triangle mesh lifetime is safe
- headless interactive target does not require OpenGL

Build/test TBB headless interactive. Build rendering targets. Compile-check CUDA paths where available.

## 9. Cleanup constraints

- Preserve core physics behavior.
- Preserve TBB/CUDA backend separation.
- No Python/NumPy dependency in interactive.
- No renderer access to `System`.
- No JSON parsing inside Session.
- No session ownership inside Transport.
- No simulation command switch inside Transport.
- Remove obsolete docs/examples that describe the old single-session in-process `Server(Session&)` architecture.
- Keep code style, ownership conventions, and one-primary-type-per-file convention used by the repository.

Finish with a concise summary of changed architecture, files removed/added, JSON schema example, and tests/builds executed.
