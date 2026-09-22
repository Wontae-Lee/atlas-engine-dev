# Expected External Client Flow

1. The client launches the native interactive executable.
2. The client establishes a future generic control transport.
3. The native process creates `Session` with its configured `SystemFactory`.
4. `Session` owns the resulting `System`; the live object never crosses the protocol.
5. The client sends `start`.
6. `Session` advances one simulation step per render loop iteration.
7. `RawStateProvider` reads the current live Fluid states.
8. `StateBridge` transfers existing particle states into reusable OpenGL buffers.
9. `ParticleLayer` draws every live particle through `Renderer` into a future `OffscreenTarget`.
10. A concrete `FrameStream` sends the rendered frame to the client.
11. The client displays the frame without receiving raw particle arrays.
12. The client may send `pause`, `step`, `status`, `save`, `restart`, and future camera-control requests.
13. The client sends `close` and `shutdown`.

A VS Code extension is one intended consumer of this frontend-neutral flow.
