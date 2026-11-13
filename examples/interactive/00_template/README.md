# 00 — Interactive template

This directory is a complete starting point for a native window backed by an
Atlas session. `config.json` initializes two particles at `(-0.5, ±0.1, 0)` with
velocity `(10, 0, 0)`, a `[-1, 1]³` domain, cell size `0.25`, and time step
`0.0001`. It has no collision solver, source, collider, or sink. Particles eventually leave the visible domain because this minimal setup has
no boundary containment.
The C++ and Python templates use the same initial state.

## Files and responsibilities

| File              | What to edit                                                                                                                                                                            |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config.json`     | Runnable strict JSON: initial state, domain, time step, and optional physical components.                                                                                               |
| `main.cpp`        | Sequential application lifecycle, camera setup, and configuration loading.                                                                                             |
| `CMakeLists.txt`  | Example name, target, configuration copy, and install rules.                                                                                                                            |
| `reference.jsonc` | Annotated API reference with alternative materials, geometry, sources, generators, sinks, and states. Copy selected fields into `config.json`; JSONC itself is not accepted at runtime. |

## Build and run

From the repository root, configure a TBB build with native rendering:

```bash
cmake -S . -B build/interactive-tbb -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_HOST_COMPILER=native \
    -DATLAS_EXAMPLES=ON -DATLAS_INTERACTIVE=ON \
    -DATLAS_INTERACTIVE_RENDERING=ON -DATLAS_PYTHON=OFF \
    -DATLAS_BENCHMARKS=OFF -DATLAS_GOOGLE_TEST=OFF -DBUILD_TESTING=OFF
cmake --build build/interactive-tbb --target atlas_example_interactive_00_template
./build/interactive-tbb/examples/interactive/00_template/atlas_example_interactive_00_template
```

OpenGL, GLEW, GLFW, and GLM are required, along with a display when running.
`ATLAS_DEVICE_SYSTEM=CUDA` selects the CUDA build; use a fresh build directory,
an nvcc-capable toolchain, and a GPU for execution. The same application source
works with either backend. A build with rendering disabled has no window example
targets; the JSONL `atlas-interactive` application can still load `config.json`.

This example follows [`02_intake`](../02_intake/README.md), the standard
Interactive example. All application steps are written directly in `main()`:
there are no helper functions, CLI options, progress output, or `try/catch`
wrappers. It runs until the window closes and then closes its session without
an additional simulation update. Inline comments explain each lifecycle stage.

Mouse drag or `W/A/S/D` orbits the camera; right drag or the wheel zooms;
`1` through `7` select fixed views; Escape closes the window.

## Create `03_my_simulation`

1. Copy the whole directory:

   ```bash
   cp -r examples/interactive/00_template examples/interactive/03_my_simulation
   ```

2. Change `ATLAS_INTERACTIVE_EXAMPLE_NAME` in the copied `CMakeLists.txt` from
   `00_template` to `03_my_simulation`. The executable and install destination
   derive from this name: `atlas_example_interactive_03_my_simulation` and
   `examples/interactive/03_my_simulation/config.json`.
   Also change the installed-directory literal in `main.cpp` to `03_my_simulation`.
3. Add `add_subdirectory(03_my_simulation)` to
   `examples/interactive/CMakeLists.txt`. There is no central case dispatcher.
   The directory's existing `add_dependencies` also includes its target in
   `atlas_examples_interactive`.
4. Edit the copied `config.json`, then update its README to state the setup and
   expected behavior. Set the initial camera with `Camera(position, target)` in
   `main.cpp`; keep physical settings in JSON.
5. Rerun the configure command and build the new target:

   ```bash
   cmake --build build/interactive-tbb --target atlas_example_interactive_03_my_simulation
   ./build/interactive-tbb/examples/interactive/03_my_simulation/atlas_example_interactive_03_my_simulation
   ```

6. Inspect the initial view and simulation behavior, then close the window.
   Use the JSONL server for bounded stepping and status queries. Keep the same directory number/name across
   C++, Interactive, and Python when they describe the same physical example.

## Write the configuration

Start with `dt`, `fluid`, and `universe`, the three required top-level fields.
Use numbers and arrays in JSON, with no comments or trailing commas.

1. **Materials and particles.** `fluid.materials` is the species table; the first
   material has species index `0`. `buffer_size` is capacity and
   `particle_count` is the live count. Each supplied `position`, `velocity`, or
   `species` column must have one entry per live particle. Positions and
   velocities have three components. Reserve enough capacity for future source
   emission. Optional energy and temperature columns also match the live count.
2. **Domain and time.** Set `universe.lower_corner`, `upper_corner`, and
   `cell_size`; a finite `geometry` can supply bounds instead. Optional Universe
   state arrays must match its derived cell count. Choose a positive `dt` suited
   to your particle speeds and geometry; the example values are a smoke setup,
   not a validated physical discretization.
3. **Emission.** Add an `emitters` entry pairing a `source` with a `generator`.
   The source has a `unit.geometry`, `spacing`, and optional `tolerance`.
   A `maxwell_boltzmann` generator uses species ratios, species numbers,
   temperature, bulk velocity, and a seed, with masses from the material table
   or explicit `species_mass`. Do not change only one species-related array when
   extending the material table.
4. **Walls and removal.** Add `colliders` for wall interactions and `sinks` for
   particles leaving the useful domain. A Unit combines local `geometry` with
   optional `translation`, scalar-first `orientation` (`[w,x,y,z]`), velocity,
   acceleration, and angular motion. The cylinder example shows six outer
   volume sinks and an isothermal wall.
5. **Collisions and analysis.** Add a `solvers` entry to enable a DSMC kernel.
   The optional `codec` selects a solver by cell using configured representative
   scales. Consult the annotated reference and the
   [Interactive field tables](../../../docs/frontends/interactive.md#simulation-json)
   before adding advanced components.

The decoder checks JSON syntax and structure. Core builders own physical and
numerical validation. Failed application responses return a nonzero exit code;
the example does not add an exception-reporting wrapper. You can also validate the JSON without opening a window through the
`validate` command of the [JSONL application](../../../docs/frontends/interactive.md#jsonl-commands).

## Understand and customize the application

`main.cpp` decodes the fixed config and sends `create`, `render_open`, and
`start` through `InteractiveApplication::handle()`. Its update loop advances
running sessions and lets `RenderManager` render when a frame is due. The
renderer never advances simulation. An initial `app.update()` draws the scene
before `start`. After the window closes, the shutdown sequence is `close` and
`shutdown`, matching `02_intake`.

`Server` owns the Session, the Session owns the Core System, and `RenderManager`
owns graphics resources. Keep those lifetimes separate when adding keyboard
commands, multiple sessions, statistics, or restart behavior. Send application
commands through `InteractiveApplication` and check every response's `success`.
For arbitrary config paths or scripted control, use the existing
`atlas-interactive --config <path>` JSONL application.

CMake copies `config.json` beside the build-tree executable with
`configure_file(COPYONLY)`. Changing the source JSON triggers CMake regeneration
on the next build, so rebuild before running an edited config. At runtime the
executable first checks beside itself, then the installed
`../examples/interactive/<number_name>/config.json`. It does not depend on the
current working directory or fall back to a source checkout. On Linux,
`/proc/self/exe` also makes invocation through `PATH` locate the executable
correctly. This example uses the same Linux executable-path lookup as `02_intake`.

Install the application and its configs with:

```bash
cmake --build build/interactive-tbb --target atlas-interactive-app atlas_examples_interactive
cmake --install build/interactive-tbb --prefix "$PWD/build/interactive-install" --component interactive
./build/interactive-install/bin/atlas_example_interactive_00_template
```

The `interactive` component installs the JSONL application as well as all
examples, so build the application and example aggregate before installing. Keep installed
`bin/` and `examples/` together
when relocating the installation.
