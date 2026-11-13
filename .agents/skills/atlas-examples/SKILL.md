---
name: atlas-examples
description: Create or change Atlas simulation examples in examples/cpp, examples/interactive, and examples/python, including their configuration, CMake targets, assets, and writing guides.
---

# Atlas examples

Follow the repository [bootstrap](../../SKILL.md) and
[common coding style](../atlas-coding-style/SKILL.md). Read the selected example's
source and README before adapting it. All three frontends use the coding style
of `02_intake`; their construction APIs remain different. The
[examples index](../../../examples/README.md) describes maintained cases and targets.

## Choose the reference

| Work | Read |
|---|---|
| Native Core construction and state access | [C++ template](../../../examples/cpp/00_template/main.cu), its [guide](../../../examples/cpp/00_template/README.md), and [cylinder](../../../examples/cpp/01_cylinder/main.cu) for boundary/solver assembly |
| Coding style for every frontend's examples | [02_intake main.cpp](../../../examples/interactive/02_intake/main.cpp); its sequential structure and readable stage comments are the standard for C++, Interactive, and Python |
| Interactive application lifecycle | [template guide](../../../examples/interactive/00_template/README.md) and `02_intake` for configuration loading, camera setup, and session commands |
| Interactive configuration fields | [reference.jsonc](../../../examples/interactive/00_template/reference.jsonc) and the [Interactive guide](../../../docs/frontends/interactive.md); JSONC is reference material, not runtime JSON |
| Python construction, engine selection, and arrays | [Python template](../../../examples/python/00_template/main.py), its [guide](../../../examples/python/00_template/README.md), and the [Python frontend guide](../../../docs/frontends/python.md) |

## Directories and build integration

- Use `NN_descriptive_name` directories and keep matching simulation numbers and
  names across frontends. `00_template` is the starting point; `02_intake` is
  reserved for intake across frontends even where its implementation is absent.
  Inspect existing directories before selecting the next number.
- Give each native example its own `CMakeLists.txt` and identifiable target:
  `atlas_example_cpp_NN_name` or `atlas_example_interactive_NN_name`. Register it
  in the frontend's `CMakeLists.txt` and add it to `atlas_examples_cpp` or
  `atlas_examples_interactive`. Do not restore a central case dispatcher or add
  a preset per example; existing presets select the backend/build type, and
  `--target` selects the example.
- For C++, follow the [template CMake file](../../../examples/cpp/00_template/CMakeLists.txt):
  keep the `.cu` source and mark it `LANGUAGE CXX` when `ATLAS_USE_NVCC` is false.
  Preserve `atlas::core` linking and the existing conditional `atlas` link.
- Interactive targets link `atlas::interactive-rendering` and are enabled only
  when execution and rendering are enabled. Copy every runtime JSON and required
  asset beside the binary with `configure_file(... COPYONLY)`. Install binaries
  to `bin` and their files under `examples/interactive/NN_name` in the
  `interactive` component. See [intake CMake](../../../examples/interactive/02_intake/CMakeLists.txt).
  When copying an example, update its installed-directory literal in `main.cpp`.
- Python examples are scripts using the installed package, without per-example
  CMake executable targets. Do not equate building `atlas_python` with installing
  the complete Python package.

## Common example standard

Use `02_intake`'s direct, sequential entry-point structure in every example:
native `main()` for C++/Interactive and the script entry point for Python.
Keep construction, execution, and shutdown visible in order. Avoid unnecessary
helper layers, CLI option parsers, diagnostic/progress printing, and exception
wrappers. Initialize values through constructors or the frontend's existing
builders, and explain stages and non-obvious choices with readable comments
above the related code. Retain necessary response/result checks.

Existing C++ and Python examples still contain helper functions, step-count CLIs,
and summary output. Those are legacy artifacts, not exceptions to the user's
current style. When creating or changing a requested example, align the affected
entry point and its documentation with this standard; do not use a narrow task
as permission for an unrelated bulk rewrite.

The common style does not merge frontend APIs. Native C++ still constructs Core
objects with builders, Python uses its binding API, and Interactive loads the
engine's JSON format. Do not introduce a Core JSON dependency to make the files
look alike. The example conventions do not govern the JSONL service's CLI and
error handling in `src/interactive/main.cpp`.

## Interactive configuration and lifecycle

Keep physics in strict JSON and application presentation in C++:

1. Resolve the example directory and read the chosen configuration directly in
   `main()`. Current examples use Linux `/proc/self/exe`, then select the build
   directory or installed example directory as the working directory.
2. Decode through `JsonCodec`. Keep flow, geometry, hinge translations, and
   angular motion in supported `SimulationConfig` fields. Use separate JSON
   files for different setups; choose the filename in `main()`. Do not inject
   example-only keys such as phase-spacing controls into engine JSON.
3. Initialize `Camera(position, target[, up])`, construct `RenderManager`, and
   pass it to `InteractiveApplication`.
4. Send `create`, address subsequent commands with its session ID, open rendering,
   and call `app.update()` once before `start` to draw the initial state. Continue
   updating while rendering is active, then send `close` and `shutdown`.

For OBJ geometry, use the engine-supported `geometry.file` and
`geometry.translation` fields. A hinge at `p` uses local mesh translation `-p`
and Unit translation `p`. Leave mesh loading and ownership to Interactive's
factory; do not convert a loaded mesh back into JSON triangle arrays in the
example. Relative asset paths resolve from the process working directory.
When changing this path, inspect both physics construction and the renderer's
borrowed mesh view. Intake's wall-clock pacing is a presentation choice, not a
required loop for every example or a change to simulation `dt`.

## C++ and Python API distinctions

Use Core builders and Atlas ownership types in C++; read device-backed particle
data through `HostBuffer` as the template demonstrates. In Python, select the
backend before importing native classes, preserve PascalCase public class names,
and use `float32` vector arrays and `uint64` species arrays. `System` consumes its
Fluid/Universe inputs; use `system.fluid` and `system.universe` afterward. Returned
particle arrays are owned snapshots, not writable views into simulation storage.

## Finish within scope

Update the example README, examples index, and affected root/frontend commands
when changing filenames, targets, assets, or runtime behavior. Distinguish
deterministic initial-state expectations from stochastic particle counts and
document intentional differences between matching frontend cases. Respect the
bootstrap's execution restrictions; changing an example does not itself authorize
a build, simulation, test, or formatter run.
