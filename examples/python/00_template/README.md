# 00 — Python simulation template

Copy this directory to start an independent Python example. `main.py` is a
complete, runnable simulation. `reference.py` preserves the broader API
construction reference: additional materials, geometries, states, generators,
colliders, sinks, solvers, and codecs. Its alternatives are construction
examples, not a validated physical setup or the code run by `main.py`.

## Install and run

Run these commands from the repository root after installing Atlas and NumPy
in your Python environment. For a local TBB installation:

```bash
python -m pip install . -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB
ATLAS_DEFAULT_ENGINE=tbb python examples/python/00_template/main.py
ATLAS_DEFAULT_ENGINE=tbb python examples/python/00_template/main.py 10
```

The optional argument is a nonnegative decimal number of complete
`System.update()` calls. It defaults to 40; `0` constructs the System and reads
its initial state. Negative numbers, fractional values, case names, and extra
arguments are rejected. Use `--help` to display the command syntax.

Python examples have independent script paths and need no CMake executable
target. The `atlas_python` target builds the binding; it does not install a
complete Python package. See the [Python frontend guide](../../../docs/frontends/python.md)
for editable installs, wheels, and CUDA installation requirements.

Select the backend before importing native classes. The environment variable
above does this before the script starts. With a combined installation, use
`ATLAS_DEFAULT_ENGINE=cuda` for CUDA and a working GPU. To choose in code, place
this before every `from atlas import ...` statement:

```python
import atlas

atlas.set_default_engine("tbb")
from atlas import Float3, Fluid, System, Universe
```

The first native import fixes the engine for that process; use separate
processes when comparing engines. Engine discovery reports installed extensions,
not GPU readiness.

## Read and customize the template

`build_system()` assembles the simulation in dependency order:

1. `MaterialDictionary` defines one molecular species. Its index is `0` in the
   particle species array. Molecular values use SI units: kilograms, joules,
   meters, and kelvin where applicable.
2. Two `float32` arrays with shape `(2, 3)` provide positions and velocities.
   A `uint64` array with shape `(2,)` provides species indices.
   `Fluid.from_arrays()` uploads owned copies. `buffer_size=1024` reserves spare
   capacity; the live particle count is 2. The statistical weight is 1 physical
   particle per simulated particle.
3. `Universe` defines the box from `(-1, -1, -1)` to `(1, 1, 1)` with a cell
   size of `0.25`, yielding 729 cells. Atlas uses
   `floor((upper - lower) / cell_size) + 1` cells per axis to cover the upper
   boundary, so this grid is `9 × 9 × 9`. A computational domain alone does not
   add a reflecting wall or an outflow sink.
4. `System` takes the Fluid and Universe and a time step of `1e-4` seconds.
   It consumes both owners. After construction, access them through
   `system.fluid` and `system.universe`; do not keep using the consumed input
   variables.

`run()` advances the System and reads the final positions. There are no
emitters, collision solver, colliders, or sinks, so the particles move at
`(10, 0, 0)` m/s. After 40 updates their x coordinate is approximately `-0.46`:

```text
template: cells=729 dt=1.0e-04s steps=40
completed: steps=40 particles=2 first_x=-0.460001
```

Use this deterministic motion to check a new environment before adding
physics. The defaults are deliberately small. If you increase the duration,
also choose a suitable domain and boundary behavior.

To extend `build_system()`, choose only the components your simulation needs:

- For emission, construct a `Unit` from geometry, create a `VolumeSource` or
  `SurfaceSource`, pair it with a generator, and pass
  `emitters=[(source, generator)]` to `System`. Species ratios, species indices,
  and material entries must describe the same population.
- For DSMC collisions, provide appropriate molecular parameters and statistical
  weight, then add `solver=DsmcSolver()`. Cell size and time step need to suit
  the physical model; a successful run does not establish numerical accuracy.
- For solid boundaries, add `IsothermalCollider(Unit(geometry), ...)` through
  `colliders`. `Sync` and optional kinematics on `Unit` describe pose and motion.
- For removal, add volume, surface, or tracing sinks through `sinks`. See
  [`01_cylinder`](../01_cylinder/README.md) for an emitter, cylinder collider,
  DSMC solver, and six domain-edge sinks working together.
- Add optional Fluid or Universe states explicitly with their state setters;
  `reference.py` contains examples and supported construction patterns.

NumPy getters return owned host snapshots. Editing `positions` after calling
`system.fluid.positions()` does not change the simulation. Use the Fluid state
setters to upload changes, and request snapshots only when needed; CUDA reads
transfer data from the device. Keep plotting, output, and analysis in the
Python application. The native Interactive renderer is a separate frontend.

## Add the next numbered example

1. Copy the directory, keeping the same number and descriptive name across
   frontends that demonstrate the same setup:

   ```bash
   cp -r examples/python/00_template examples/python/02_my_simulation
   ```

2. Edit `02_my_simulation/main.py`: set the physical inputs in `build_system()`,
   add the required components, and adapt `run()` and its summary to your output.
   Keep the direct `if __name__ == "__main__"` entry point. No case dispatcher or
   parent registry needs editing.
3. Replace this README with the purpose, assumptions, units, dependencies,
   execution command, expected output, and tuning instructions for the example.
   Remove `reference.py` from the copied directory if it is no longer useful.
4. Run the script directly with a short step count and with `0`, then check the
   default run. Confirm particle counts stay within capacity and state values
   remain finite. Add a focused integration check for behavior the example
   promises, and update the [example index](../../README.md).

```bash
ATLAS_DEFAULT_ENGINE=tbb python examples/python/02_my_simulation/main.py 5
ATLAS_DEFAULT_ENGINE=tbb python -X faulthandler -m unittest discover \
    -s tests/python -p test_integration.py -v
```
