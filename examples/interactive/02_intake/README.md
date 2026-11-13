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

The example has no command-line options. `main()` loads `config_2.json` and
runs until the window closes. Change that filename to `config_1.json` and rebuild
to run synchronized motion. Camera setup and display pacing are directly in
`main.cpp`; simulation settings remain in the two JSON files.

## Two simulation cases

Both cases use identical geometry, flow, amplitude, smoothing, and period.

- **Case 1 — synchronized:** all twelve wings start at phase zero and close and
  open together, matching the original example.
- **Case 2 — traveling wave:** the twelve wings sample one complete wave around
  the ring. Each neighbor is separated by `1 / 12` of a cycle, including the
  wrap from wing 12 back to wing 1. Every wing keeps the same period and
  amplitude; the crest moves around the circumference without a phase seam.

For case 2, wing `i` has initial elapsed time
`period * (i - 1) / 12`, stored in each wing’s `angular_oscillation.elapsed`. These are initial phase offsets, so the first
frame already shows a wave. With the defaults, wing 1 starts at 0 degrees,
wing 7 at 20 degrees, and wing 12 at approximately 1.34 degrees. Restart restores
this wave.

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

Each JSON sets geometry `translation` to the negative pivot and the Unit's
`translation` to the pivot. Thus zero rotation reproduces the original OBJ assembly
exactly, while positive rotation closes each wing toward the axis. Geometry is
loaded once; only rigid poses change during stepping. These are prescribed
hinges, not a mechanical linkage model or a solid-contact simulation. Wings do
not resolve contact with one another; use conservative angles when changing the
setup.

## Motion and flow settings

Edit `config_1.json` for synchronized motion or `config_2.json` for the
traveling wave, then rebuild to copy the updated files beside the executable.
Both files use only engine `SimulationConfig` fields. Collider 0 is the fixed
support; colliders 1 through 12 correspond to `wing1.obj` through `wing12.obj`.
All hinge positions and motion parameters are explicit in these files; selecting
a case only chooses the file and does not calculate or override motion settings.

| Setting | Default | Meaning |
|---------|---------|---------|
| `colliders[i].unit.geometry.file` | `assets/support.obj`, `assets/wing1.obj` … | OBJ asset path |
| `colliders[i].unit.geometry.translation` | Negative hinge position | Local mesh coordinate correction |
| `colliders[i].unit.translation` | Radius `0.25`, height `0.05` | Hinge position in meters |
| `colliders[i].unit.angular_oscillation.axis` | Tangent to the ring | Rotation axis |
| `colliders[i].unit.angular_oscillation.amplitude` | `0.34906585` | Maximum inward angle in radians (20 degrees) |
| `colliders[i].unit.angular_oscillation.period` | `0.02` | Simulation seconds per cycle |
| `colliders[i].unit.angular_oscillation.smooth` | `true` | Smooth acceleration and deceleration |
| `colliders[i].unit.angular_oscillation.elapsed` | Case 1: `0`; case 2: `0.02 * (i - 1) / 12` | Initial time within the cycle |
| `dt` | `0.00005` | 400 updates per cycle |
| `fluid.buffer_size` | `20000` | Maximum live particle count |
| `emitters[0].generator.bulk_velocity` | `[0,-500,0]` | Inflow velocity in m/s |
| `emitters[0].source.spacing` | `0.25` | Inflow seed spacing in meters |
| `universe.cell_size` | `0.15` | Spatial grid size in meters |

Display pacing remains `display_seconds_per_cycle = 5` in `main.cpp`; it does
not change simulation physics. The default cases share a period, and pacing
uses the first wing's configured period.

With `angular_oscillation.smooth = true` (the example default), the prescribed angle is:

```text
theta(t) = A/2 * (1 - cos(2*pi*t/T))
```

In case 1, every wing starts at rest at the original pose, reaches 20 degrees at step 200, returns
to the original pose at step 400, then repeats. Rotation slows smoothly to zero
at both ends.
Floating-point rounding can affect the final digits; stochastic flow counts
are not exact reference values.

Set each wing’s `angular_oscillation.smooth` in the selected JSON file to `false` for constant-speed closing and opening. With
`phase = (t mod T) / T`, its angle is `2*A*phase` during the first half-cycle
and `2*A*(1-phase)` during the second. It reaches the same endpoints at steps
200 and 400, but reverses direction immediately without slowing down. Angular
velocity is `+2*A/T` while closing and `-2*A/T` while opening; the reversal uses
the velocity of the new half-cycle. The reported angular acceleration is zero
between reversals; the idealized instantaneous reversal has no finite
acceleration value.

Core `Unit` owns the angular trajectory and its instantaneous angular velocity.
Moving-wall reflection therefore uses the same motion as the displayed mesh.
Session stepping advances the simulation, and the renderer only observes its
current pose. Pausing a Session freezes motion; restarting restores each wing's
configured initial phase and pose. Motion is based on simulation time,
independent of frame rate.

`main.cpp` sets the working directory to the example directory and decodes the
selected JSON using `JsonCodec`. Both JSON files contain the OBJ paths, local
mesh translations, Unit poses, and motion settings. Interactive's factory loads meshes
through `atlas::TriangleMesh::load_from_obj()`; the example has no mesh loader or
triangle-array conversion. The same JSON can be loaded by the generic server
when its working directory is the example directory with the copied `assets/`.
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
./build/interactive-install/bin/atlas_example_interactive_02_intake
```

The install includes this executable, both case configs, and its OBJ/MTL files under
`examples/interactive/02_intake/assets`. Keep the installed `bin/` and
`examples/` directories together when relocating it.
