# Atlas Interactive Examples

## JSON server

`simulation.json` is a runnable configuration for the headless
`atlas-interactive` process. Build and run the TBB server from the repository
root:

```bash
cmake -S . -B build/interactive-headless-tbb -G Ninja \
    -DATLAS_DEVICE_SYSTEM=TBB \
    -DATLAS_INTERACTIVE=ON \
    -DATLAS_INTERACTIVE_RENDERING=OFF \
    -DATLAS_PYTHON=OFF \
    -DATLAS_EXAMPLES=OFF
cmake --build build/interactive-headless-tbb --target atlas-interactive-app
./build/interactive-headless-tbb/src/interactive/atlas-interactive \
    --config examples/interactive/simulation.json
```

The process emits a startup response containing session ID 1. Enter requests
from `external_client/requests.jsonl` one line at a time, or pipe a command:

```bash
printf '%s\n' \
    '{"request_id":"status-1","session_id":1,"command":"status"}' \
    '{"request_id":"shutdown-1","command":"shutdown"}' |
./build/interactive-headless-tbb/src/interactive/atlas-interactive \
    --config examples/interactive/simulation.json
```

Stdout contains JSONL responses only. Native Atlas logs use stderr.

## Native window

`native_window.cpp` is the local rendering reference. It constructs a
`SimulationConfig`, owns the resulting simulation through `Session`, and
composes `RawStateProvider`, `Renderer`, `ParticleLayer`,
`GeometryLayer`, and `WindowTarget`. Each loop iteration advances exactly
one step and renders the new raw particle and boundary state.

```bash
cmake --preset tbb-application-release
cmake --build build/tbb-application-release --target atlas-interactive-example
./build/tbb-application-release/examples/interactive/atlas-interactive-example
```

Pass a positive step count to close automatically:

```bash
./build/tbb-application-release/examples/interactive/atlas-interactive-example 10
```

Drag with the left mouse button or use `W`, `A`, `S`, and `D` to orbit.
Drag with the right mouse button, use the scroll wheel, or press `Q` and `E`
to zoom. Number keys 1 through 7 select the front, back, top, bottom, left,
right, and perspective views.

The same source builds with `ATLAS_DEVICE_SYSTEM=CUDA`. Backend transfer stays
inside `StateBridge`; the example contains no CUDA/OpenGL resource management.
CUDA uses direct interop when OpenGL and CUDA use the same NVIDIA GPU, with a
host-staged fallback for hybrid-GPU configurations.

See [the interactive guide](../../docs/guidelines/interactive.md) for the
configuration schema, protocol, ownership, and backend transfer paths.
