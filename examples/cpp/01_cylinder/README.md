# 01 — Native C++ cylinder flow

`main.cu` independently constructs and runs a molecular freestream around an
open cylinder. There is no case selector or dependency on the template's entry
point. The target is `atlas_example_cpp_01_cylinder`.

Configure native examples as shown in the
[template guide](../00_template/README.md), then run from the repository root:

```bash
cmake --build build/examples-tbb --target atlas_example_cpp_01_cylinder
./build/examples-tbb/examples/cpp/01_cylinder/atlas_example_cpp_01_cylinder 40
```

The optional nonnegative decimal step count defaults to 40. `0` constructs and
reports without advancing; `--help` shows usage. Invalid or extra arguments
produce an error. The executable name identifies the simulation.

## Model and editing points

| Setting in `run()`                | Default                                  |
|-----------------------------------|------------------------------------------|
| Domain lower / upper (m)          | `(-1, -0.75, -0.5)` / `(1.5, 0.75, 0.5)` |
| Cell size / timestep              | `0.1` m / `5e-5` s                       |
| Freestream velocity / temperature | `(500, 0, 0)` m/s / `300` K              |
| Cylinder radius / height          | `0.2` m / `1.0` m, open ends             |
| Particle capacity / initial count | `20000` / `0`                            |
| Statistical weight                | `3.236e16`                               |

The volume source emits from an upstream box using a Maxwell–Boltzmann
velocity generator. A DSMC solver handles particle collisions, an isothermal
collider handles the cylinder surface, and six thin volume sinks remove
particles near the domain faces. The code prints live particle counts every ten
steps and at the final step, then copies the live position/velocity prefix into
host buffers to report the downstream population and mean streamwise velocity.

Change the local parameters and builder calls in `run()` to explore the model.
Keep capacity, source spacing, domain size, sink thickness, and timestep
consistent with the revised flow. The setup is a smoke example, not a validated
engineering benchmark. Random collision sampling and backend differences mean
final particle counts and velocities should not be treated as exact fixtures.

The matching Python and Interactive examples are
[Python 01_cylinder](../../python/01_cylinder/README.md) and
[Interactive 01_cylinder](../../interactive/01_cylinder/README.md).
