# Frontend boundaries

```text
Python binding ───────→ Atlas Core ←────── Interactive
```

Atlas Core owns simulation data, numerical policies, and the ordered update
pipeline. It has no Python, JSON, application output, or graphics dependency.
The two frontends use Core directly and do not depend on each other.

The [Python binding](../frontends/python.md) exposes Core objects with a
PascalCase API. It adapts Python lifetimes and type keyed C++ state access and
returns owned NumPy snapshots. Engine selection occurs before native imports.

Interactive is the native control and presentation layer:

```text
JsonLinesTransport → JsonCodec → InteractiveApplication
                                  ├─ Server → Session → CoreFactory → Core builders → System
                                  └─ RenderManager → Renderer → WindowTarget
                                                        ↑
                                 Session::scene_view() → RawStateProvider → StateBridge
```

`CoreFactory`, also exposed by the compatibility name `SystemFactory`, translates
`SimulationConfig` into Core builder calls. The decoder owns JSON structure
checks; the builders own Atlas validity rules. `Server` stores multiple sessions
and routes simulation commands. Each `Session` owns exactly one live `System`,
its run state, statistics, and optional CSV output. It exposes temporary,
read-only state and geometry views for rendering but owns no OpenGL resources.

`InteractiveApplication` coordinates commands across the server and optional
`RenderManager`. The manager selects one existing session and owns the window and
renderer. `Renderer` draws a fresh scene view without advancing simulation;
`Server::update()` advances running sessions. Closing a window only releases
rendering resources. The session can continue running, be paused, stepped, or
rendered again. Closing its session or shutting down closes its renderer first.

The execution library remains usable without the graphics library, and the
headless build has no OpenGL dependency. See the [Interactive guide](../frontends/interactive.md)
for commands, state transfer, and rendering details.
