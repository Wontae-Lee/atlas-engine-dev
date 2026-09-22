Work in the `Wontae-Lee/atlas-engine-dev` repository on `feature/engine-interactive`, based on the latest `main`.

At the time of this plan, `main` is at:

```text
71ceea1 feat(interactive): restore controls and support hybrid GPUs
```

First inspect the current `main` implementation and preserve working behavior while restructuring the interactive layer.

# Goal

Turn `src/interactive` into the execution/control layer for an Atlas simulation.

The immediate target is to run a complete simulation locally using only the interactive layer, including:

* simulation lifecycle control
* continuous execution
* pause and manual stepping
* status reporting
* simulation statistics
* per-source particle generation counts
* per-sink particle removal counts
* CSV statistics output
* simulation snapshot saving
* restarting a simulation from its initial configuration
* optional rendering

The architecture must also remain suitable for a future remote server/workstation/HPC deployment.

The central architectural rule is:

```text
Session is the sole interactive-layer gateway to atlas-core.
```

`Session` must be the only interactive class that owns and accesses the live `atlas::System`.

It is also the only class that interprets and executes simulation commands.

The high-level architecture should become:

```text
External Client / Local Frontend
              │
              │ Request
              ▼
            Server
              │
              │ forward
              ▼
            Session
      ┌────────┼──────────────┐
      │        │              │
      │        ▼              ▼
      │ SimulationStatistics  CsvWriter
      │
      ▼
 unique ownership of
   atlas::System
      │
      ▼
   atlas-core
```

Rendering uses data exposed by `Session`, not `atlas::System` directly:

```text
atlas::System
      │
      │ accessed only by Session
      ▼
    Session
      │
      ▼
SimulationRenderView
      │
      ▼
   Renderer
      │
      ▼
 StateBridge
      │
      ▼
 RenderTarget
```

---

# 1. Refactor Session into the central simulation object

Current code:

```text
src/interactive/session/session.h
src/interactive/session/session.cpp
```

currently stores:

```cpp
std::optional<System> _system;
```

and publicly exposes:

```cpp
System& system();
const System& system() const;
```

Refactor this.

Use Atlas's unique ownership type for the live system:

```cpp
atlas::SystemHostPtr _system;
```

Verify the existing Atlas ownership alias and use the existing project convention rather than introducing a competing pointer type.

`Session` owns exactly one live `System`.

Suggested structure:

```text
session/
├── session.h
├── session.cpp
├── session_state.h
├── session_status.h
├── session_config.h
└── system_factory.h
```

Keep one class/struct/enum per file where practical.

## SessionState

Move the current nested state enum into its own file:

```cpp
enum class SessionState {
    empty,
    ready,
    running,
    paused
};
```

## Session

`Session` should become responsible for:

```text
live System ownership
simulation lifecycle
command dispatch
simulation execution
status extraction
statistics sampling
snapshot saving
simulation restart
render-view creation
shutdown request state
```

A target API can look conceptually like:

```cpp
class Session final {
public:
    explicit Session(SystemFactory factory, SessionConfig config = {});

    Response handle(const Request& request);

    void update();

    SessionStatus status() const;
    const SimulationStatistics& statistics() const noexcept;

    SimulationRenderView render_view() const;

    bool shutdown_requested() const noexcept;

private:
    void initialize();
    void start();
    void pause();
    void step(std::size_t count);
    void advance_once();

    void save(const std::filesystem::path& path);
    void restart();
    void close();

    SimulationSample collect_sample() const;

private:
    SystemFactory _factory;
    atlas::SystemHostPtr _system;

    SessionState _state = SessionState::empty;

    SimulationStatistics _statistics;
    std::unique_ptr<CsvWriter> _csv_writer;

    bool _shutdown_requested = false;
};
```

Adjust exact signatures where necessary to fit the existing project style.

The important ownership hierarchy is:

```text
Session
 ├── SystemFactory
 ├── unique System
 ├── SimulationStatistics
 ├── optional CsvWriter
 └── SessionState
```

The live `System` must not escape through a public `system()` accessor.

All data needed outside Session must instead be exposed as interactive-layer value/view types such as:

```text
SessionStatus
SimulationStatistics
SimulationRenderView
Response
```

---

# 2. Add restart support through a C++ SystemFactory

A live `System` is move-only and uniquely owned, so restarting from the initial configuration requires a way to create a fresh system.

Introduce a simple C++ factory type, for example:

```cpp
using SystemFactory = std::function<atlas::SystemHostPtr()>;
```

Place it in:

```text
session/system_factory.h
```

A Session receives and stores this factory.

Initialization:

```text
SystemFactory
    ↓
Session
    ↓
factory()
    ↓
unique atlas::System
```

Restart:

```text
Command::restart
      ↓
Session::handle()
      ↓
Session::restart()
      ↓
destroy current System
reset statistics
reset/reopen CSV output
call SystemFactory
own fresh System
state = ready
```

This gives true "restart from the initial C++ configuration" without trying to clone `System`.

Use the existing C++ `System::builder()` APIs inside the application-supplied factory.

Update `src/interactive/main.cpp` and the interactive example so their system creation functions return `SystemHostPtr`, preferably using:

```cpp
System::builder()
    ...
    .make_host_unique();
```

---

# 3. Move all command interpretation into Session

Current command handling lives in:

```text
src/interactive/transport/server.cpp
```

and `Server::handle()` contains the `switch(request.command)`.

Move command interpretation into:

```cpp
Session::handle(const Request&)
```

The only command switch should belong to Session.

The Session command flow should be:

```text
Request
   ↓
Session::handle()
   ├── start
   ├── pause
   ├── step
   ├── status
   ├── save
   ├── restart
   ├── close
   └── shutdown
```

The command set for this phase should support at least:

```cpp
enum class Command {
    start,
    pause,
    step,
    status,
    save,
    restart,
    close,
    shutdown
};
```

`initialize` does not need to remain a protocol command for the current C++ implementation because construction/reconstruction is handled through `SystemFactory`.

Extend `Request` with the data needed by these commands, including:

```text
step_count
filesystem path for save when required
```

Keep the protocol simple and C++-native for now.

---

# 4. Make Server a thin server/forwarding layer

Move:

```text
transport/server.h
transport/server.cpp
```

to:

```text
server/server.h
server/server.cpp
```

`Server` represents the server-side boundary, which will later allow a local frontend, workstation client, TCP transport, or HPC client to reach the Session.

Its responsibility is intentionally narrow:

```text
receive Request
      ↓
forward Request to Session
      ↓
receive Response
      ↓
send/return Response
```

Conceptually:

```cpp
Response
Server::handle(const Request& request) {
    return _session->handle(request);
}
```

or an equivalent serving loop once transport is connected.

Server may also observe:

```cpp
session.shutdown_requested()
```

to terminate its serving loop.

Server should not implement the meaning of `start`, `pause`, `step`, `save`, etc.

That behavior belongs to Session.

---

# 5. Move protocol out of transport

Current:

```text
transport/
├── server.*
├── frame_stream.*
└── protocol/
    ├── command.h
    ├── request.h
    └── response.h
```

Restructure to:

```text
protocol/
├── command.h
├── request.h
└── response.h

server/
├── server.h
└── server.cpp

transport/
├── frame_stream.h
├── frame_stream.cpp
└── future transport implementations
```

Protocol is independent of a specific transport mechanism.

This boundary should later allow:

```text
local transport
TCP transport
HPC transport/tunnel
```

to carry the same Request/Response objects without changing Session.

---

# 6. Add SessionStatus

Create:

```text
session/session_status.h
```

SessionStatus should represent the lightweight externally visible simulation state.

At minimum include:

```cpp
struct SessionStatus final {
    SessionState state = SessionState::empty;

    std::size_t step = 0;
    double simulation_time = 0.0;

    std::size_t particle_count = 0;

    std::size_t source_count = 0;
    std::size_t sink_count = 0;
};
```

Session builds this object by reading its private `_system`.

No other interactive class should query `System` to construct status.

`Response` should be extended so a status command can return a `SessionStatus`, for example through an optional payload.

---

# 7. Implement SimulationStatistics entirely in interactive

Create:

```text
statistics/
├── simulation_sample.h
├── simulation_statistics.h
└── simulation_statistics.cpp
```

## SimulationSample

This represents one completed simulation step.

Suggested contents:

```cpp
struct SimulationSample final {
    std::size_t step = 0;
    double simulation_time = 0.0;

    std::size_t particle_count = 0;

    std::vector<std::size_t> source_spawned;
    std::vector<std::size_t> sink_removed;
};
```

## SimulationStatistics

This class owns current and cumulative statistics.

It receives only `SimulationSample`.

Example responsibility:

```text
current sample
total particles spawned per source
total particles removed per sink
completed sample count
statistics reset on restart
```

Conceptually:

```cpp
class SimulationStatistics final {
public:
    void update(const SimulationSample& sample);
    void reset();

    const SimulationSample& current() const noexcept;

    const std::vector<std::uint64_t>& total_source_spawned() const noexcept;
    const std::vector<std::uint64_t>& total_sink_removed() const noexcept;

private:
    SimulationSample _current;

    std::vector<std::uint64_t> _total_source_spawned;
    std::vector<std::uint64_t> _total_sink_removed;
};
```

The data flow must be:

```text
atlas::System
     ↓
   Session
     ↓
SimulationSample
     ↓
SimulationStatistics
```

`SimulationStatistics` should have no dependency on `atlas::System`.

---

# 8. Add minimal raw event counters to atlas-core

Statistics remain in interactive, but two pieces of information currently exist only inside `System`:

```text
particles spawned by each Source during the last step
particles removed by each Sink during the last step
```

Add only the minimum raw event information needed to `atlas::System`.

Do not add accumulated run statistics to core.

## Source counters

Current `System::emit()` already computes:

```cpp
const int spawned = _sources[i]->spawn(positions, count);
```

Add a last-step source count buffer to `System`.

A suitable representation is a host-side integer buffer sized to `_sources.size()`.

At every emit phase:

```text
reset all source counters to zero
for every source:
    execute spawn
    store the actual spawned value at the same source index
```

Expose a read-only last-step accessor usable by Session.

Suggested semantic API:

```cpp
source_spawned_last_step()
```

## Sink counters

Current `System::mark_survivors()` already determines exactly which sink first claims each particle:

```cpp
for (int s = 0; s < sink_count; ++s) {
    if (sinks[s].despawn(...)) {
        active[i] = 0;
        return;
    }
}
```

Add a per-sink last-step removal counter.

Because `mark_survivors()` runs through `ExecutionPolicy::device`, implement this using a backend-compatible device counter buffer.

The repository already provides:

```cpp
atlas::atomic_add()
```

in:

```text
include/atlas/parallel/atomic.h
```

Use it when a sink claims a particle:

```text
first matching sink
      ↓
atomic increment sink counter[s]
      ↓
active[i] = 0
      ↓
return
```

Use a counter type supported consistently by the existing TBB/CUDA backend. Since the current pipeline particle count is already bounded by `int`, a `DeviceBuffer<int>` is a practical last-step counter representation.

At the beginning of each remove phase, reset all sink counters to zero.

Provide a host-readable accessor for Session. A bulk device-to-host copy of the small per-sink counter array is acceptable when Session collects one statistics sample after a completed step.

The resulting core responsibility is only:

```text
What happened during the most recent physical step?
```

Interactive remains responsible for:

```text
accumulation
averages
flux calculations
CSV
reporting
run-level statistics
```

---

# 9. Centralize step execution in Session::advance_once()

Implement one private method that represents one complete interactive simulation step:

```cpp
Session::advance_once()
```

Its conceptual flow should be:

```text
_system->update()
      ↓
collect SimulationSample from _system
      ↓
_statistics.update(sample)
      ↓
append CSV row if CSV recording is enabled
```

`Session::update()`:

```text
if state == running:
    advance_once()
```

`Session::step(count)`:

```text
repeat count times:
    advance_once()
```

This ensures manual stepping and continuous execution produce identical:

```text
statistics
source/sink counters
CSV output
step/time tracking
```

and avoids duplicating post-step logic.

---

# 10. Add streaming CSV output

Create:

```text
output/
├── csv_writer.h
└── csv_writer.cpp
```

`CsvWriter` receives interactive-layer statistics objects, never `atlas::System`.

The CSV should record at least:

```text
step
simulation_time
particle_count
each source's particles spawned during this step
each sink's particles removed during this step
```

Example:

```csv
step,time,particle_count,source_0_spawned,source_1_spawned,sink_0_removed,sink_1_removed
100,0.001000,51200,125,120,93,18
101,0.001010,51332,130,119,98,19
```

Cumulative columns may also be added if useful:

```text
source_0_total
sink_0_total
...
```

Use streaming append rather than retaining all historical samples in memory.

Expected flow:

```text
advance_once()
    ↓
SimulationSample
    ↓
SimulationStatistics
    ↓
CsvWriter::append(...)
```

Generate the CSV header from the actual number of sources and sinks when a Session initializes.

Add a small `SessionConfig` if needed to define:

```text
output directory
CSV enabled/disabled
CSV file name
```

Session owns the CsvWriter lifecycle.

Restart should reset statistics and start a fresh run output according to the chosen SessionConfig behavior.

---

# 11. Add save support through Session

The existing core already provides:

```cpp
System::save(directory)
```

which writes:

```text
time_step_<step>/
├── fluid.bin
└── universe.bin
```

Expose this through Session command handling:

```text
Command::save
     ↓
Session::handle()
     ↓
Session::save(path)
     ↓
_system->save(path)
```

Flush CSV output as part of save so numerical statistics and the physical snapshot are consistent from the user's perspective.

Keep restart and save as separate operations:

```text
save
    persist current simulation state

restart
    destroy current live System
    build a fresh System from SystemFactory
    reset interactive run statistics/output
    return to ready state
```

A later full checkpoint/resume mechanism can build on this once all System-owned moving/configuration state is serializable.

---

# 12. Refactor rendering so Session remains the sole core gateway

Current rendering directly depends on `atlas::System`:

```cpp
Renderer::render(const System&, RenderTarget&)
RawStateProvider::update(const System&)
```

and `main.cpp` currently calls:

```cpp
renderer.render(session.system(), target);
```

Refactor this boundary.

Create backend-native, non-owning rendering views, for example:

```text
view/
├── simulation_buffer_view.h
└── simulation_render_view.h
```

## SimulationBufferView

This should describe existing native memory without copying it.

Conceptually:

```cpp
struct SimulationBufferView final {
    const void* data = nullptr;
    std::size_t bytes = 0;
};
```

Add additional metadata only where required by rendering.

## SimulationRenderView

Represent the particle buffers needed by the current renderer:

```text
particle count
position buffer
velocity buffer
species buffer
optional temperature
optional energy buffers
```

`Session::render_view()` reads the private System/Fluid and constructs these non-owning views.

The buffers remain owned by the Atlas core objects inside Session.

The render view only points to them temporarily.

Document the lifetime clearly: the view is valid while the current Session System and its state buffers remain unchanged; callers should obtain a fresh view after simulation updates that may mutate/reallocate state.

---

# 13. Preserve the CUDA direct rendering path

The rendering refactor must preserve the current efficient CUDA/OpenGL path.

For a CUDA build on the same NVIDIA GPU as OpenGL, the intended data path remains:

```text
atlas CUDA DeviceBuffer
        │
        │ native device pointer
        ▼
Session::render_view()
        │
        │ non-owning pointer only
        ▼
RawStateProvider / StateBridge
        │
        │ CUDA/OpenGL interop
        │ device-to-device copy
        ▼
OpenGL Buffer
        │
        ▼
Renderer / WindowTarget
```

There should be no mandatory host copy introduced by the Session boundary.

Refactor `StateBridge` so it can upload from the non-owning buffer view rather than requiring direct access to an Atlas `DeviceBuffer<T>`.

The backend-specific StateBridge implementation already knows whether this build is TBB or CUDA:

```text
TBB:
native host pointer
    ↓
OpenGL upload

CUDA:
native device pointer
    ↓
CUDA/OpenGL interop
    ↓
device-to-device OpenGL buffer upload
```

Preserve the existing CUDA fallback behavior for systems where CUDA and OpenGL use different GPUs or direct interop registration is unavailable:

```text
CUDA device
    ↓
host staging
    ↓
OpenGL GPU
```

That remains a fallback path rather than the normal CUDA rendering path.

Refactor:

```cpp
RawStateProvider::update(...)
Renderer::render(...)
```

to consume `SimulationRenderView` instead of `atlas::System`.

After this change, rendering code should no longer require direct knowledge of System, Fluid, or Universe.

---

# 14. Update main.cpp to demonstrate the new architecture

The current `main.cpp` accesses:

```cpp
session.system().step()
renderer.render(session.system(), target)
```

Replace this with the Session-facing API.

The reference usage should look conceptually like:

```cpp
atlas::interactive::Session session(
    [] {
        return make_system();
    },
    config);

atlas::interactive::Server server(session);

session.handle({ .command = Command::start });

while (...) {
    session.update();

    const SessionStatus status = session.status();

    const SimulationRenderView view = session.render_view();
    renderer.render(view, target);
}
```

Adapt exact request construction to the final Request API.

`make_system()` should return an owning `SystemHostPtr`.

Once the System is created for Session, all later simulation operations flow through Session.

---

# 15. Split execution/control from rendering at the CMake target level

Current `src/interactive/CMakeLists.txt` makes OpenGL/GLEW/GLFW/GLM mandatory for the whole interactive library.

Restructure this so the simulation execution/server layer can build on a headless machine or HPC compute node.

Use two logical targets.

## Interactive execution target

For example:

```text
atlas::interactive
```

Contains:

```text
Session
Server
Protocol
SimulationStatistics
CsvWriter
views that have no OpenGL dependency
```

Links to:

```text
atlas core
```

It should be buildable without:

```text
OpenGL
GLEW
GLFW
GLM
```

## Interactive rendering target

For example:

```text
atlas::interactive-rendering
```

Contains:

```text
Renderer
Camera
StateProvider
StateBridge
Layers
OpenGL Buffer/Shader/Framebuffer
WindowTarget
OffscreenTarget
```

Links to:

```text
atlas::interactive
OpenGL
GLEW
GLFW
GLM
```

CUDA-specific StateBridge compilation remains selected with the existing:

```text
ATLAS_DEVICE_SYSTEM
```

mechanism.

This gives the future deployment model:

```text
Headless server / HPC:
atlas core
+ atlas::interactive

Local visualization:
atlas core
+ atlas::interactive
+ atlas::interactive-rendering
```

---

# 16. Preserve class/file ownership clarity

Keep the repository structure easy to understand by directory and filename.

A target structure is:

```text
src/interactive/
│
├── session/
│   ├── session.h
│   ├── session.cpp
│   ├── session_state.h
│   ├── session_status.h
│   ├── session_config.h
│   └── system_factory.h
│
├── server/
│   ├── server.h
│   └── server.cpp
│
├── protocol/
│   ├── command.h
│   ├── request.h
│   └── response.h
│
├── statistics/
│   ├── simulation_sample.h
│   ├── simulation_statistics.h
│   └── simulation_statistics.cpp
│
├── output/
│   ├── csv_writer.h
│   └── csv_writer.cpp
│
├── view/
│   ├── simulation_buffer_view.h
│   └── simulation_render_view.h
│
├── transport/
│   ├── frame_stream.h
│   └── frame_stream.cpp
│
├── rendering/
│   ├── camera.*
│   ├── renderer.*
│   ├── backend/
│   ├── state/
│   ├── layer/
│   ├── target/
│   └── opengl/
│
└── main.cpp
```

Use one primary class/struct/enum per file.

Objects that exist only as part of one owning class should be stored and managed by that owner rather than exposed as global/shared state.

---

# 17. Tests and verification

Add tests around the new architecture.

At minimum verify:

### Session lifecycle

```text
factory creates System
→ ready

start
→ running

update
→ exactly one System step

pause
→ update no longer advances

step(10)
→ exactly ten simulation steps are executed

restart
→ old System destroyed
→ fresh System created
→ step resets
→ statistics reset
→ state becomes ready

close
→ live System released
→ state becomes empty
```

### Command routing

Verify that:

```text
Server forwards Request
Session handles command
Response reflects Session result
```

### Statistics

Use a deterministic System with known source/sink behavior and verify:

```text
source_spawned current counts
source_spawned cumulative counts
sink_removed current counts
sink_removed cumulative counts
particle count consistency
step/time consistency
```

### Core event counters

Verify source counters are reset every step.

Verify sink counters count the first sink that claims each particle exactly once.

Verify TBB behavior and ensure CUDA compilation remains valid.

### CSV

Verify:

```text
header
source/sink dynamic columns
one row per completed step
restart behavior
save flush behavior
```

### Rendering

Verify that the native window example still renders after Renderer is converted from:

```cpp
System&
```

to:

```cpp
SimulationRenderView
```

For CUDA, preserve the direct CUDA/OpenGL interop implementation in `StateBridge`.

---

# Implementation order

Implement in this order so each stage leaves the tree coherent:

```text
1. SessionState / SessionStatus / protocol relocation
2. Session unique System ownership + SystemFactory
3. move command dispatch from Server into Session
4. thin Server forwarding layer
5. Session::advance_once()
6. core source last-step counters
7. core sink last-step counters
8. SimulationSample / SimulationStatistics
9. CsvWriter + SessionConfig output
10. save command
11. restart command
12. SimulationBufferView / SimulationRenderView
13. refactor RawStateProvider / Renderer away from System
14. update native main/example
15. split CMake execution and rendering targets
16. tests and documentation
```

After implementation, build and run the relevant TBB interactive target and tests. Also perform a CUDA compile/build check where available, paying particular attention to the sink atomic counters and the refactored StateBridge.

The final architecture should make this true:

```text
External simulation command
        ↓
      Server
        ↓
      Session
        ↓
 unique live System
        ↓
     Atlas Core
```

and:

```text
Atlas Core data
        ↓
      Session
     /       \
statistics   render view
    ↓            ↓
CSV          Renderer
```

This Session-centered boundary is the foundation for the later workstation/HPC server implementation, while the current branch should already be capable of running, controlling, observing, saving, restarting, recording, and optionally rendering a local simulation.
