# 02 — Dynamic intake wings

This example loads `assets/intake/wing1.obj` through `wing12.obj` and
`support.obj` and repeatedly folds all twelve wings inward and opens them.
Case 1 preserves the original synchronized motion. Case 2 uses the same period
and amplitude with a continuous traveling wave around the ring. The support stays
fixed. A molecular inflow enters from +Y toward -Y, interacting with the moving
mesh colliders; six outer sinks remove outgoing particles.

## Build and run

From the repository root, with the dependencies in the
[build guide](../../../docs/contributing/build.md) installed:

```bash
cmake --preset tbb-release
cmake --build --preset tbb-release --target atlas_example_interactive_02_intake
./build/tbb-release/examples/interactive/02_intake/atlas_example_interactive_02_intake
```

Replace `tbb-release` with `cuda-release` for CUDA. Each source mesh is copied
beside the executable during configuration. Source asset or config changes are
picked up when CMake regenerates on the next build. No working-directory or
source-checkout lookup is required at runtime.

The default opens a window and repeats until you close it. One complete cycle
is paced to about five wall-clock seconds if computation can keep up; slower
machines take longer. Mouse drag or `W/A/S/D` orbits; right drag or the wheel
zooms. Press `3` for a view into the intake along its Y axis, or `7` for an
oblique view. Escape closes the window. The initial camera frames the entire
intake; edit the Camera setup in `main.cpp` to choose another initial view.

```bash
# Two complete cycles, paced for viewing.
./build/tbb-release/examples/interactive/02_intake/atlas_example_interactive_02_intake 800

# Case 1: the original synchronized motion (also the default).
./build/tbb-release/examples/interactive/02_intake/atlas_example_interactive_02_intake 800 --case 1 --headless

# Case 2: a traveling wave with the same period, paced for viewing.
./build/tbb-release/examples/interactive/02_intake/atlas_example_interactive_02_intake --case 2

# Two case-2 cycles without display or wall-clock pacing.
./build/tbb-release/examples/interactive/02_intake/atlas_example_interactive_02_intake 800 --case 2 --headless

# Bounded windowed smoke run at computational speed.
./build/tbb-release/examples/interactive/02_intake/atlas_example_interactive_02_intake 40 --unpaced
```

The optional step count limits total updates exactly: there is no extra step
on shutdown in this example. Omission or `0` means unlimited windowed execution.
`--headless` requires a positive step count and never opens a window. It still
uses this example's rendering-enabled build target; the underlying simulation
and motion have no graphics dependency. `--case 1` is the default; `--case 2`
selects the traveling wave. The case option can appear before or after the step
count and accepts only 1 or 2. `--help` prints argument syntax.

## Two simulation cases

Both cases use identical geometry, flow, amplitude, smoothing, and period.

- **Case 1 — synchronized:** all twelve wings start at phase zero and close and
  open together, matching the original example.
- **Case 2 — traveling wave:** the twelve wings sample one complete wave around
  the ring. Each neighbor is separated by `1 / 12` of a cycle, including the
  wrap from wing 12 back to wing 1. Every wing keeps the same period and
  amplitude; the crest moves around the circumference without a phase seam.

For case 2, wing `i` has initial elapsed time
`period_seconds * (i - 1) / 12`. These are initial phase offsets, so the first
frame already shows a wave. With the defaults, wing 1 starts at 0 degrees,
wing 7 at 20 degrees, and wing 12 at approximately 1.34 degrees. Restart restores
this wave. Console reports label `wing1` and `wing12` separately.

This makes the angular motion continuous around the ring. The OBJ wings remain
independent rigid meshes; this does not weld their edges or deform the meshes.

## Initial geometry and hinges

OBJ headers specify meters. `intake.obj` is the combined assembly and is not
loaded in addition to the individual parts. There are 8,040 triangles in each
wing and 824 in the support, for 97,304 triangles total. MTL files accompany the
OBJ files; the current renderer colors by boundary role rather than CAD material.

The CAD export contains no hinge metadata. This example models each hinge at
the top of the support ring, using radius `0.25 m` and height `0.05 m`. For wing
`i` from 1 to 12, with `phi = (i - 1) * pi / 6`:

```text
pivot = (radius * sin(phi), height, radius * cos(phi))
axis  = (-cos(phi), 0, sin(phi))
```

The loader subtracts the pivot from mesh vertices and puts it back as the
Unit's translation. Thus zero rotation reproduces the original OBJ assembly
exactly, while positive rotation closes each wing toward the axis. Geometry is
loaded once; only rigid poses change during stepping. These are prescribed
hinges, not a mechanical linkage model or a solid-contact simulation. Wings do
not resolve contact with one another; use conservative angles when changing the
setup.

## Motion and flow settings

Edit engine settings in `config.json` and example motion/display settings in
`main.cpp`, then rebuild before running:

| Setting | Default | Meaning |
|---|---|---|
| `closing_angle_degrees (main.cpp)` | `20` | Maximum inward angle relative to the original assembly |
| `period_seconds (main.cpp)` | `0.02` | Physical simulation time for close + reopen, shared by every wing in both cases |
| `smooth_motion` in `main.cpp` | `true` | Smooth acceleration and deceleration; `false` moves at constant angular speed between the endpoints |
| `display_seconds_per_cycle (main.cpp)` | `5` | Wall-clock pacing target; does not change physics |
| `hinge_radius (main.cpp)`, `hinge_height (main.cpp)` | `0.25`, `0.05` | Modeled hinge location in meters |
| `dt` | `0.00005` | 400 updates per cycle |
| `fluid.buffer_size` | `20000` | Maximum live particle count |
| `emitters[0].generator.bulk_velocity` | `[0,-500,0]` | Inflow velocity in m/s |
| `emitters[0].source.spacing` | `0.25` | Inflow seed spacing in meters |
| `universe.cell_size` | `0.15` | Spatial grid size in meters |

With `smooth_motion = true` (the example default), the prescribed angle is:

```text
theta(t) = A/2 * (1 - cos(2*pi*t/T))
```

In case 1, every wing starts at rest at the original pose, reaches 20 degrees at step 200, returns
to the original pose at step 400, then repeats. Rotation slows smoothly to zero
at both ends. Console reports print half-cycle angles and particle counts.
Floating-point rounding can affect the final digits; stochastic flow counts
are not exact reference values.

Set `smooth_motion` in `main.cpp` to `false` for constant-speed closing and opening. With
`phase = (t mod T) / T`, its angle is `2*A*phase` during the first half-cycle
and `2*A*(1-phase)` during the second. It reaches the same endpoints at steps
200 and 400, but reverses direction immediately without slowing down. Angular
velocity is `+2*A/T` while closing and `-2*A/T` while opening; the reversal uses
the velocity of the new half-cycle. The reported angular acceleration is zero
between reversals; the idealized instantaneous reversal has no finite
acceleration value. The console angles follow the selected motion mode and case-specific phases.

Core `Unit` owns the angular trajectory and its instantaneous angular velocity.
Moving-wall reflection therefore uses the same motion as the displayed mesh.
Session stepping advances the simulation, and the renderer only observes its
current pose. Pausing a Session freezes motion; restarting restores each wing's
configured initial phase and pose. Motion is based on simulation time,
independent of frame rate.

`config.json` contains only the engine's `SimulationConfig` fields. Case
selection, phase spacing, hinge geometry, degree conversion, and display pacing
live in `main.cpp`. The executable adds mesh colliders using the engine's
`angular_oscillation` fields (`axis`, `amplitude`, `period`, `smooth`, `elapsed`)
before decoding the simulation. Loading the JSON directly into the generic
server loads the base flow configuration without these example-created wings.
The engine JSON format is documented in the
[Interactive guide](../../../docs/frontends/interactive.md#simulation-json).

For a geometry-only inspection, set `emitters` and
`solvers` to empty arrays. The initial Fluid has zero particles, so
the same moving collision geometry remains visible without inflow. For flow
experiments, tune the timestep, particle weight, emission density, grid, and
motion period together. The defaults demonstrate coupling and are not a
validated engineering benchmark.

## Install

```bash
cmake --build --preset tbb-release --target atlas-interactive-app atlas_examples_interactive
cmake --install build/tbb-release --prefix "$PWD/build/interactive-install" --component interactive
./build/interactive-install/bin/atlas_example_interactive_02_intake 800 --headless
```

The install includes this executable, its config, and its OBJ/MTL files under
`examples/interactive/02_intake/assets`. Keep the installed `bin/` and
`examples/` directories together when relocating it.
