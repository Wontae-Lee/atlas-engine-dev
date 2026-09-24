# Atlas Examples

The examples present the same Atlas simulation through three consumers of the
core: native C++, Python, and the native Interactive application. Each frontend has
one entry point, a reusable reference template, and focused runnable cases.

```text
examples/
├── cpp/
│   ├── main.cu
│   ├── template.h
│   └── cases/cylinder.h
├── interactive/
│   ├── main.cpp
│   ├── template.jsonc
│   └── cases/cylinder.json
└── python/
    ├── main.py
    ├── template.py
    └── cases/cylinder.py
```

`main.*` selects a case and handles command-line arguments. Files under
`cases/` are small simulations intended to run. The `template.*` files show the
broader construction surface, including alternatives that a single physical
case does not need. Copy a template when exploring the API; start from a case
when building a runnable simulation.

## Cylinder flow

All three frontends default to `cylinder`. The case emits a molecular
freestream, advances it around an open cylinder, and removes particles that
leave the domain. The numerical setup is intentionally small enough for a smoke
run and is not a validated engineering benchmark.

## Development environment

Build the standard image with `python3 scripts/dev.py build tbb`, then enter
`python3 scripts/dev.py tbb`. The commands below run inside that shell with
preinstalled dependencies. Use the `cuda` image for GPU execution. To open an
Interactive window, enter with `ATLAS_DOCKER_DISPLAY=1 python3 scripts/dev.py tbb`
(or `cuda`) so the host X11 display is available. See
[Docker development](../docs/operations/docker.md#standard-development-environment).

## Native C++

Enable examples in a TBB build and run the shared example executable:

```bash
cmake -S . -B build/examples-tbb -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DATLAS_DEVICE_SYSTEM=TBB \
    -DATLAS_EXAMPLES=ON \
    -DATLAS_PYTHON=OFF \
    -DATLAS_INTERACTIVE=OFF
cmake --build build/examples-tbb --target atlas_example
./build/examples-tbb/examples/cpp/atlas_example cylinder 40
```

Use `ATLAS_DEVICE_SYSTEM=CUDA` with an nvcc-capable toolchain for the CUDA
variant. A GPU is required when the executable runs.

## Native Interactive window

Interactive execution and rendering must both be enabled. The example loads a
JSON case, creates a session through `InteractiveApplication`, opens rendering
with `render_open`, and starts the session through the same command path as the
JSONL server. Application updates advance running sessions; `RenderManager`
draws frames independently. It displays live particles and configured source,
collider, and sink geometry at their current poses.

```bash
cmake -S . -B build/interactive-tbb -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DATLAS_DEVICE_SYSTEM=TBB \
    -DATLAS_INTERACTIVE=ON \
    -DATLAS_INTERACTIVE_RENDERING=ON \
    -DATLAS_EXAMPLES=ON \
    -DATLAS_PYTHON=OFF
cmake --build build/interactive-tbb --target atlas-interactive-example
./build/interactive-tbb/examples/interactive/atlas-interactive-example cylinder 40
```

The first argument may also be an explicit JSON file. Omitting the second
argument runs until the window closes. The optional number limits simulation
steps, rather than frames. After the window closes, the example pauses,
closes rendering, steps the still-live session once, and then closes it. Mouse
drag or `W/A/S/D` orbits the
camera, right drag or the scroll wheel zooms, keys `1` through `7` select fixed
views, and Escape closes the window.

`template.jsonc` documents every accepted configuration branch. Runnable case
files use strict JSON because the same decoder accepts them at runtime.

## Python

Install an Atlas wheel, select the engine before importing native classes when
needed, and run:

```bash
ATLAS_DEFAULT_ENGINE=tbb python examples/python/main.py cylinder 40
```

The Python case mirrors the native setup and reads final Fluid state as owned
NumPy snapshots for application-side analysis.

## Adding a case to every frontend

1. Add `examples/cpp/cases/<name>.h` with a `run()` function and dispatch it
   from `examples/cpp/main.cu`.
2. Add `examples/python/cases/<name>.py` with a `run()` function and dispatch it
   from `examples/python/main.py`.
3. Add `examples/interactive/cases/<name>.json`. The Interactive entry point
   finds it automatically by case name; no C++ dispatch change is required.
4. Keep the three configurations physically equivalent where their public APIs
   permit it, and keep case defaults small enough for a smoke run.

Frontend architecture and protocol details belong in
[`docs/frontends/`](../docs/frontends/), while build and validation policy
belongs in [`docs/contributing/`](../docs/contributing/).
