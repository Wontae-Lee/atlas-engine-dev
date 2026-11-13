# Atlas Examples

Each frontend uses numbered, independent example directories. `00_template` is
an executable starting point with a detailed writing guide; `01_cylinder` is the
maintained freestream-over-cylinder DSMC simulation. Select the executable or
Python script directly. There is no shared case-selection entry point.

```text
examples/
├── cpp/
│   ├── 00_template/     # main.cu, CMakeLists.txt, README.md, reference.h
│   └── 01_cylinder/     # main.cu, CMakeLists.txt, README.md
├── interactive/
│   ├── 00_template/     # main.cpp, config.json, CMakeLists.txt, README.md, reference.jsonc
│   ├── 01_cylinder/     # main.cpp, config.json, CMakeLists.txt, README.md
│   └── 02_intake/       # moving OBJ wings, config.json, main.cpp, README.md
└── python/
    ├── 00_template/     # main.py, README.md, reference.py
    └── 01_cylinder/     # main.py, README.md
```

The templates initialize two particles moving at 10 m/s in an otherwise empty
domain. They introduce construction, stepping, and state access without a
collision model. The `reference.*` files preserve the broader API/configuration
catalogue; copy only the alternatives needed for your simulation. They are not
validated physical scenarios. The cylinder examples emit a molecular
freestream, reflect it from an open cylinder, and remove particles near domain
boundaries. Their defaults are smoke runs, not engineering benchmarks.

## Targets and entry points

| Frontend | Template | Cylinder |
|---|---|---|
| C++ CMake target | `atlas_example_cpp_00_template` | `atlas_example_cpp_01_cylinder` |
| Interactive CMake target | `atlas_example_interactive_00_template` | `atlas_example_interactive_01_cylinder` |
| Python script | `python/00_template/main.py` | `python/01_cylinder/main.py` |

`atlas_examples_cpp` builds both C++ programs. `atlas_examples_interactive`
builds all windowed programs when rendering is enabled. `atlas_examples`
builds every enabled native example. Python scripts import the installed Atlas
package and need no per-example CMake target.

## Native C++

From the repository root:

```bash
cmake --preset tbb-release
cmake --build --preset tbb-release --target atlas_examples_cpp
./build/tbb-release/examples/cpp/00_template/atlas_example_cpp_00_template 40
./build/tbb-release/examples/cpp/01_cylinder/atlas_example_cpp_01_cylinder 40
```

To build only the cylinder, use
`cmake --build --preset tbb-release --target atlas_example_cpp_01_cylinder`.
Replace `tbb` with `cuda` in the configure/build preset and build directory for
nvcc/CUDA. Running CUDA executables requires a GPU. Native TBB builds need no
CUDA toolkit. Full presets also configure Python, benchmarks, and rendering;
install their [dependencies](../docs/contributing/build.md) first, or use the
narrow configure command in the template guide.

The sole positional argument is the nonnegative step count (default 40).
Zero constructs the System and reports its initial state without advancing it.
Invalid or extra arguments fail instead of silently running another simulation.
See the [C++ template writing guide](cpp/00_template/README.md).

## Native Interactive window

```bash
cmake --preset tbb-release
cmake --build --preset tbb-release --target atlas_examples_interactive
./build/tbb-release/examples/interactive/00_template/atlas_example_interactive_00_template 40
./build/tbb-release/examples/interactive/01_cylinder/atlas_example_interactive_01_cylinder 40
```

These presets enable execution, rendering, and examples. OpenGL, GLEW, GLFW,
GLM, and a graphics display are required. For a single program, use
`cmake --build --preset tbb-release --target atlas_example_interactive_01_cylinder`.
Use `cuda-release` for the CUDA backend.

The template and cylinder executables each load their own `config.json`; they
do not accept a case name.
Omit the step limit (or pass zero) to run until the window closes. The positive
limit counts simulation updates, not rendered frames. After the window loop,
the example pauses the session, closes rendering, performs one additional
headless step to demonstrate independent lifetimes, and closes the session.
Mouse drag or `W/A/S/D` orbits the camera, right drag or the wheel zooms,
`1` through `7` select fixed views, and Escape closes the window.

Use the JSONL server for arbitrary configurations or display-free execution:

```bash
./build/tbb-release/src/interactive/atlas-interactive \
    --config examples/interactive/01_cylinder/config.json
```

Build that server separately with `--target atlas-interactive-app`. The
[Interactive template writing guide](interactive/00_template/README.md)
explains strict JSON, configuration copying, installation, application/session
ownership, and headless commands. `reference.jsonc` contains comments and is
not accepted as runtime JSON.

## Dynamic intake (Interactive)

[`interactive/02_intake`](interactive/02_intake/README.md) loads the twelve intake
wings and support from `assets/intake`. The wings repeatedly close inward by
20 degrees and return to their original assembly, with moving-wall particle
collisions. Build `atlas_example_interactive_02_intake` and run it without
arguments for the synchronized case-1 window, or with `800 --headless` for two
complete cycles. Pass `--case 2` for the same period with slightly staggered wing
phases; `motion.phase_spread` controls the phase difference around the ring.
Its guide explains both cases, the inferred hinges, motion/flow settings, and
asset install.
This example is currently implemented only for Interactive.

## Python

Install the Atlas package as described in the
[Python frontend guide](../docs/frontends/python.md), then run:

```bash
ATLAS_DEFAULT_ENGINE=tbb python examples/python/00_template/main.py 40
ATLAS_DEFAULT_ENGINE=tbb python examples/python/01_cylinder/main.py 40
```

Select the engine before importing native classes. Both scripts accept only an
optional nonnegative step count, defaulting to 40. Fluid reads return owned
NumPy snapshots. See the [Python template writing guide](python/00_template/README.md)
for initialization, ownership, state output, and adding components.

## Writing the next example

1. Copy the frontend's `00_template` directory to the next unused number,
   currently `03_<descriptive_name>`. `02_intake` is reserved for the matching
   intake setup in every frontend. Keep matching numbers and names across
   frontends when they demonstrate the same simulation.
2. Implement the setup in that directory's `main.cu`, `main.py`, or
   `config.json`, using its template guide. Keep a local README explaining
   purpose, parameters, commands, and expected output.
3. For C++ and Interactive, rename the target in the copied `CMakeLists.txt`,
   register the directory in the frontend's `CMakeLists.txt`, and add the target
   to its frontend aggregate. Reconfigure CMake. No dispatcher needs editing.
4. Keep the existing backend/build-type presets. Select the new executable
   with `cmake --build --preset tbb-release --target atlas_example_<frontend>_03_<name>`;
   the aggregate targets include it through its local `add_dependencies` call.
5. Build and run the new example and retained template/cylinder smoke runs.
   Keep common configurations physically equivalent where the APIs permit it.

Previous `atlas_example` / `atlas-interactive-example` targets and
`main.* cylinder 40` commands have been replaced by the explicit targets and
paths above. Update local IDE run configurations accordingly.

Architecture and protocol details live in [`docs/frontends/`](../docs/frontends/).
Build and test policy lives in [`docs/contributing/`](../docs/contributing/).
