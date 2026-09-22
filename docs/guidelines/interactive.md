# Interactive Execution and Rendering

`src/interactive/` is the native C++ execution, control, and rendering layer for
Atlas. It consumes the Atlas core directly. The core does not depend on it, and
the Python package is a separate consumer:

```text
Python binding ───────→ Atlas Core ←────── Interactive
```

The interactive layer supports a headless execution/control library and an
optional OpenGL renderer. It contains no Python, NumPy, or frontend-specific
particle path.

## Responsibility boundaries

| Component | Responsibility |
|---|---|
| `Session` | Own the live `System`, execute commands, step it, collect run statistics, save snapshots, restart, and expose render views. |
| `SystemFactory` | Build a fresh move-only `System` for initial construction and restart. |
| `Server` | Forward a `Request` to `Session`; it contains no command switch or simulation logic. |
| `SimulationStatistics` | Hold the latest sample and cumulative source/sink counts for the current run. |
| `CsvWriter` | Stream statistics supplied by `Session` to a caller-configured file. |
| `SimulationRenderView` | Describe the current live particle buffers without transferring or owning them. |
| `StateProvider` | Convert a render view into graphics-side `RenderState`. |
| `StateBridge` | Transfer core backend memory into reusable OpenGL buffers. |
| `Renderer` | Orchestrate a provider, camera, layers, and target without stepping the simulation. |
| `Layer` | Draw one aspect of the current `RenderState`. |
| `RenderTarget` | Own where OpenGL output is presented. |

`Session` is the only interactive class that owns or accesses the live
`atlas::System`. It does not expose a public `system()` accessor. Rendering and
future transports consume interactive value or view types instead.

The current native loop is deliberately synchronous:

```text
Session::update()
    ↓
Session::render_view()
    ↓
RawStateProvider
    ↓
StateBridge
    ↓
OpenGL Buffer
    ↓
ParticleLayer
    ↓
WindowTarget
```

Each rendered frame therefore represents the raw state after one completed
simulation step.

## Building

Enable the execution/control target with `ATLAS_INTERACTIVE`. Rendering is a
separate option so a headless process does not need OpenGL, GLEW, GLFW, or GLM.

Build the TBB execution/control library and its tests without graphics:

```bash
cmake -S . -B build/interactive-headless-tbb -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DATLAS_DEVICE_SYSTEM=TBB \
    -DATLAS_INTERACTIVE=ON \
    -DATLAS_INTERACTIVE_RENDERING=OFF \
    -DATLAS_GOOGLE_TEST=ON \
    -DATLAS_PYTHON=OFF \
    -DATLAS_EXAMPLES=OFF \
    -DATLAS_BENCHMARKS=OFF
cmake --build build/interactive-headless-tbb --target atlas-interactive atlas_tests_interactive
ctest --test-dir build/interactive-headless-tbb -R '^atlas_tests_interactive\.' --output-on-failure
```

Build the native TBB renderer and example with its focused preset:

```bash
cmake --preset tbb-application-release
cmake --build build/tbb-application-release \
    --target atlas-interactive-app atlas-interactive-example
```

The library targets are `atlas::interactive` for headless execution/control and
`atlas::interactive-rendering` for OpenGL rendering. The installed executable is
`atlas-interactive`. The source-tree example target produces
`examples/interactive/atlas-interactive-example`.

Use `cuda-application-release` for the CUDA variant. It requires nvcc and the CUDA
toolkit. Running it requires a compatible NVIDIA driver and GPU.

## Creating and controlling a session

A `System` is move-only, so a session receives a factory that can build a fresh
instance. The same factory is called again by `restart`.

```cpp
#include "protocol/command.h"
#include "protocol/request.h"
#include "server/server.h"
#include "session/session.h"

#include <atlas/atlas.h>

atlas::SystemHostPtr make_system() {
    auto fluid = atlas::Fluid::builder()
        .with_buffer_size(1024)
        .with_particle_count(0)
        .make_host_unique();

    auto universe = atlas::Universe::builder()
        .with_lower_corner(atlas::Float3(-1.0f))
        .with_upper_corner(atlas::Float3(1.0f))
        .with_cell_size(0.1f)
        .make_host_unique();

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(1.0e-5f)
        .make_host_unique();
}

atlas::interactive::Session session(make_system);
atlas::interactive::Server server(session);

server.handle({atlas::interactive::Command::start});
while (!server.shutdown_requested()) {
    session.update();
}
```

`Session::update()` advances once only while the session is running. The
`step` command advances its requested count regardless of the ready or paused
state. All completed steps use the same internal path, so statistics and CSV
output are consistent between continuous and manual stepping.

Commands have these meanings:

| Command | Behavior |
|---|---|
| `start` | Enter `running`; subsequent `Session::update()` calls advance. |
| `pause` | Enter `paused`; `Session::update()` becomes a no-op. |
| `step` | Execute `Request::step_count` steps immediately. |
| `status` | Return current state, step, time, particle count, and source/sink counts. |
| `save` | Flush CSV and call core snapshot persistence at `Request::path`. |
| `restart` | Destroy the current system, call the factory, reset statistics/output, and enter `ready`. |
| `close` | Release the live system and enter `empty`. |
| `shutdown` | Set the server-loop shutdown request flag. |

`Response::success` reports whether the command completed. Invalid lifecycle or
configuration operations return a failed response with the exception message.

## Statistics, CSV, and snapshots

The core exposes only raw last-phase source and sink counters. `Session`
converts them into a `SimulationSample`, and `SimulationStatistics` owns current
and cumulative values for the active run.

Enable CSV recording with `SessionConfig`:

```cpp
atlas::interactive::SessionConfig config;
config.csv_enabled = true;
config.output_directory = "results";
config.csv_filename = "statistics.csv";

atlas::interactive::Session session(make_system, config);
```

The writer opens the configured file for the run, derives per-source and
per-sink columns from the created `System`, and appends one row after every
completed step. Columns include step, simulation time, particle count, current
spawn/removal counts, and cumulative counts. It does not retain historical rows
in memory.

`restart` starts a fresh run: statistics reset and the configured CSV is
reopened from a new header. `save` is separate; it flushes CSV before writing
`time_step_<step>/fluid.bin` and `universe.bin` through `System::save()`.

## Rendering

The native composition is:

```cpp
atlas::interactive::Session session(make_system);
atlas::interactive::Renderer renderer(
    std::make_unique<atlas::interactive::RawStateProvider>());
renderer.add_layer(std::make_unique<atlas::interactive::ParticleLayer>());

atlas::interactive::WindowTarget target(1280, 720, "Atlas");
renderer.initialize(target);

session.handle({atlas::interactive::Command::start});
while (!target.should_close()) {
    target.poll_events(renderer.camera());
    session.update();
    renderer.render(session.render_view(), target);
}

renderer.shutdown();
session.handle({atlas::interactive::Command::close});
```

`SimulationRenderView` aliases the session-owned native buffers. It is valid
only until the next operation that may resize, compact, destroy, or restart the
system. Obtain a fresh view immediately before each render and never retain it.
Required position, velocity, and species buffers are always present; optional
temperature and energy buffers appear only when the Fluid owns those states.

For TBB, `StateBridge` uploads directly from the core CPU buffer into the
OpenGL buffer. For CUDA, it persistently registers reusable OpenGL buffers and
uses a device-to-device copy through CUDA/OpenGL interop. Registration is
recreated only when a graphics buffer grows. If the active OpenGL device cannot
support interop, the implementation falls back to a host-staged upload instead
of silently failing. The normal interoperable path has no device-to-host
particle readback.

Run the native example, optionally limiting it to a fixed number of steps:

```bash
./build/tbb-application-release/examples/interactive/atlas-interactive-example
./build/tbb-application-release/examples/interactive/atlas-interactive-example 10
```

Press Escape to close. Mouse and keyboard camera controls are documented in
[`examples/interactive/README.md`](../../examples/interactive/README.md).

## Docker

The `tbb` and `cuda` images are complete Atlas runtimes. They contain Python,
NumPy, the selected Atlas engine, examples, assets, native executables, and the
graphics runtime libraries required by those executables.

```bash
docker build --target tbb -t atlas:tbb .
docker build --target cuda -t atlas:cuda .
```

They need access to a display server and graphics device only when opening the
native window. Invoke `atlas-interactive` explicitly; the default command
remains Python. CUDA additionally needs `--gpus all` and NVIDIA
graphics/display driver capabilities. See [docker.md](docker.md) for the image
matrix.

## Current limits

`InterpolationStateProvider`, `OffscreenTarget`, `FrameStream`, and the generic
external-client `Server` transport are architectural boundaries only. There is
no interpolation algorithm, headless OpenGL context, encoder, network listener,
or wire-format parser yet. Protocol commands remain ordinary C++ values.

Future offscreen clients will receive rendered frames through `FrameStream`.
They will not receive raw particle arrays through Python lists, NumPy, JSON, or
frontend-owned GPU uploads.
