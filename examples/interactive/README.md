# Atlas Interactive Examples

## Native window

`native_window.cpp` is the working reference path. It creates a deterministic
64-particle simulation, gives its `SystemFactory` to `Session`, and composes
`RawStateProvider`, `Renderer`, `ParticleLayer`, and `WindowTarget`. Each loop
iteration advances exactly one simulation step and renders all live particles
from that new raw state. Drag with the left mouse button or use `W`, `A`, `S`,
and `D` to orbit the camera. Drag with the right mouse button, use the scroll
wheel, or press `Q` and `E` to zoom. Number keys `1` through `7` select the
front, back, top, bottom, left, right, and perspective views.

Configure and build the TBB example from the repository root:

```bash
cmake --preset tbb-application-release
cmake --build build/tbb-application-release --target atlas-interactive-example
./build/tbb-application-release/examples/interactive/atlas-interactive-example
```

Pass a positive step count to close automatically after a smoke run:

```bash
./build/tbb-application-release/examples/interactive/atlas-interactive-example 10
```

The same source builds with `ATLAS_DEVICE_SYSTEM=CUDA`. Backend-specific memory
transfer remains inside `StateBridge`; the example contains no CUDA or OpenGL
resource-management code. CUDA builds use direct CUDA/OpenGL interop when both
APIs use the same NVIDIA GPU and fall back to a host-staged upload when the
window is rendered by another GPU.

## External client and offscreen architecture

The generic control concepts (`Command`, `Request`, `Response`, and `Server`)
exist. `Session` handles start, pause, step, status, save, restart, close, and
shutdown requests, but no transport serialization or listener is implemented.
`FrameStream` is an abstract interface. `OffscreenTarget` reserves the output
boundary but does not yet own a headless OpenGL context, so it is not runnable.

The planned path reuses the same `Session`, `RawStateProvider`, `Renderer`, and
layers with `OffscreenTarget`. A future `FrameStream` implementation will send
rendered frames to an arbitrary client. Particle arrays remain inside native
Atlas and OpenGL memory; they are not sent through the control protocol.

See `external_client/` for the proposed request sequence and responsibility
flow. A VS Code extension can be one consumer, but no frontend-specific code is
part of this repository.

The complete execution/control and rendering guide is
[`docs/guidelines/interactive.md`](../../docs/guidelines/interactive.md).
