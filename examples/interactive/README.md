# Atlas Interactive Examples

## Native window

`native_window.cpp` is the working reference path. It creates a deterministic
64-particle simulation, gives the `System` to `Session`, and composes
`RawStateProvider`, `Renderer`, `ParticleLayer`, and `WindowTarget`. Each loop
iteration advances exactly one simulation step and renders all live particles
from that new raw state.

Configure and build the TBB example from the repository root:

```bash
cmake -S . -B build/interactive-tbb -G Ninja \
    -DATLAS_DEVICE_SYSTEM=TBB \
    -DATLAS_INTERACTIVE=ON \
    -DATLAS_EXAMPLES=ON \
    -DATLAS_PYTHON=OFF \
    -DATLAS_GOOGLE_TEST=OFF \
    -DATLAS_BENCHMARKS=OFF
cmake --build build/interactive-tbb --target atlas-interactive-example
./build/interactive-tbb/examples/interactive/atlas-interactive-example
```

Pass a positive step count to close automatically after a smoke run:

```bash
./build/interactive-tbb/examples/interactive/atlas-interactive-example 10
```

The same source builds with `ATLAS_DEVICE_SYSTEM=CUDA`. Backend-specific memory
transfer remains inside `StateBridge`; the example contains no CUDA or OpenGL
resource-management code.

## External client and offscreen architecture

The generic control concepts (`Command`, `Request`, `Response`, and `Server`)
exist, but no transport serialization or listener is implemented. `FrameStream`
is an abstract interface. `OffscreenTarget` reserves the output boundary but
does not yet own a headless OpenGL context, so it is not runnable.

The planned path reuses the same `Session`, `RawStateProvider`, `Renderer`, and
layers with `OffscreenTarget`. A future `FrameStream` implementation will send
rendered frames to an arbitrary client. Particle arrays remain inside native
Atlas and OpenGL memory; they are not sent through the control protocol.

See `external_client/` for the proposed request sequence and responsibility
flow. A VS Code extension can be one consumer, but no frontend-specific code is
part of this repository.
