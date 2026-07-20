# Python Bindings

Atlas ships an optional [nanobind](https://nanobind.readthedocs.io) extension
module, `atlas`, that mirrors the C++ builder API so a full DSMC simulation can
be assembled, stepped, and read back from Python. The engine itself is unchanged;
the bindings live entirely under [`src/python/atlas/`](../../src/python/atlas/)
and link `atlas::core`.

## Building the module

The module is built when `ATLAS_PYTHON` is on. It needs the nanobind submodule
and a Python 3.8+ interpreter with development headers:

```bash
git submodule update --init --recursive external/nanobind
cmake -S . -B build/tbb -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_PYTHON=ON -DATLAS_BENCHMARKS=OFF
cmake --build build/tbb --target atlas_python
```

The compiled artifact imports as `atlas`:

```bash
PYTHONPATH=build/tbb/src/python/atlas python -c "import atlas"
```

Array read-back returns numpy arrays, so `numpy` is a declared runtime dependency
(`pyproject.toml`) and is installed with the wheel; the extension still imports
without it, but `positions()`, `velocities()`, `species()`, and
`fluid_from_arrays()` need it.

## Packaging a wheel and installing it later

For a redistributable, self-contained wheel, build inside the `wheel` Docker stage
(it pre-installs scikit-build-core, nanobind, build, auditwheel, and patchelf) and
run [`scripts/build_wheels.sh`](../../scripts/build_wheels.sh). Initialise the
submodules the build needs first — the script builds against the mounted sources:

```bash
git submodule update --init --recursive \
    external/nanobind external/tinyobj external/protobuf

docker build --target wheel -t atlas-wheel .

# CPU-only wheel (no GPU or driver at run time); omit ATLAS_WHEEL_BACKENDS for both.
docker run --rm -v "$PWD":/workspace -w /workspace \
    -e ATLAS_WHEEL_BACKENDS=TBB \
    atlas-wheel -c 'bash scripts/build_wheels.sh'
```

auditwheel repairs the wheel — bundling `libtbb`, `libstdc++`, and (for CUDA)
`libcudart`, but never the `libcuda.so.1` driver — and writes it to `dist/tbb/`
(and `dist/cuda/` when both backends are built). The module is compiled with
nanobind's `STABLE_ABI`: on Python 3.12+ this yields one `abi3` wheel that serves
later versions, but on earlier interpreters the wheel is version-specific (e.g.
`cp311`), so build one per target Python. Install and use it:

```bash
pip install dist/tbb/atlas-0.1.0-*.whl
python -c "import atlas; print(atlas.build_system)"
```

The TBB wheel carries its own TBB runtime, so no system `libtbb` is required. A
CUDA wheel additionally needs a matching NVIDIA driver and a GPU at run time. To
build a wheel without Docker, install the front-end tools
(`pip install "scikit-build-core>=0.10" "nanobind>=2.0" build`) and run
`python -m build --wheel -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB`, but such a wheel
is not auditwheel-repaired and depends on the host's `libtbb`.

## Quick start

```python
import atlas
import numpy as np

Vec = atlas.Float3

materials = atlas.material_dictionary([
    atlas.molecule(mass=4.65e-26, translational_energy=0.0, rotational_energy=0.0,
                   vibrational_energy=0.0, reference_diameter=4.17e-10,
                   reference_temperature=273.0, viscosity_index=0.74,
                   scattering_parameter=1.0),
])

rng = np.random.default_rng(0)
positions = rng.uniform(0.1, 0.9, size=(200, 3)).astype(np.float32)
velocities = (rng.standard_normal((200, 3)) * 300.0).astype(np.float32)

fluid = atlas.fluid_from_arrays(positions, velocities, statistical_weight=1e18, materials=materials)
universe = atlas.universe(Vec(0, 0, 0), Vec(1, 1, 1), cell_size=1.0)
solver = atlas.dsmc_solver(kernel_type=atlas.DsmcKernelType.variable_hard_sphere)

system = atlas.build_system(fluid=fluid, universe=universe, dt=1e-4, solver=solver)
for _ in range(20):
    system.update()

print(system.particle_count, np.linalg.norm(system.velocities(), axis=1).mean())
```

A runnable version is [`examples/python/dsmc_dense_cell.py`](../../examples/python/dsmc_dense_cell.py).

## The API

Every C++ builder is exposed as a **factory function** that runs the builder and
returns a ready object; value primitives are classes with constructors. Names are
lower-case; the objects they return are opaque handles you pass on to the next
factory.

### Math and geometry

| Call | Returns |
|---|---|
| `Float3(x, y, z)` / `Quaternion(w, x, y, z)` / `Quaternion(axis, radians)` | value types |
| `sphere(center, radius)` | `Geometry` |
| `plane(normal, offset)` | `Geometry` |
| `box(lower, upper)` | `Geometry` |
| `cylinder(center, radius, height, open=False)` | `Geometry` |
| `circle(center, normal, radius)` | `Geometry` |
| `square(center, normal, side_length)` | `Geometry` |
| `triangle(a, b, c, normal=None)` | `Geometry` |
| `polygonal_prism(center, side_count, radius, height)` | `Geometry` |
| `triangle_mesh(path)` | `Geometry` loaded from a Wavefront OBJ file |

`Float3` supports `+`, `-`, `* float`, indexing, and `len`.

### Transform, material, and the two state owners

| Call | Returns |
|---|---|
| `sync(translation, orientation=Quaternion())` | `Sync` (a rigid pose) |
| `unit(geometry, sync=None, velocity=None, angular_velocity=None, acceleration=None, angular_acceleration=None)` | `Unit` (a placed body) |
| `molecule / atom / ion / neutron(mass, translational_energy, rotational_energy, vibrational_energy, reference_diameter, reference_temperature, viscosity_index, scattering_parameter)` | `Material` |
| `solid(mass)` | `Material` |
| `material_dictionary([Material, ...])` | `MaterialDictionary` |
| `fluid(buffer_size, particle_count=0, statistical_weight=1.0, materials=None)` | `Fluid` |
| `fluid_from_arrays(positions, velocities, statistical_weight=1.0, materials=None)` | `Fluid` seeded from `(N, 3)` arrays |
| `universe(lower, upper, cell_size)` | `Universe` |
| `universe_from_geometry(geometry, cell_size)` | `Universe` |

`Fluid` exposes `particle_count` and `buffer_size`; `Universe` exposes `cell_count`.

### Emitter, solver, and boundaries

| Call | Returns |
|---|---|
| `volume_source(unit, spacing, tolerance=0.0)` / `surface_source(...)` | `Source` |
| `maxwell_boltzmann_generator(species_ratios, species_numbers, materials, temperature, bulk_velocity, seed)` | `Generator` |
| `uniform_generator(species_ratios, species_numbers, temperature, min_value, max_value, bulk_velocity, seed)` | `Generator` |
| `jittering_generator(species_ratios, species_numbers, temperature, base_value, jitter_radius, bulk_velocity, seed)` | `Generator` |
| `maxwell_sigma_generator(species_ratios, species_numbers, temperature, sigma, bulk_velocity, seed)` | `Generator` |
| `dsmc_solver(kernel_type=DsmcKernelType.variable_hard_sphere, majorant_sample_pairs=8, majorant_exhaustive_limit=5)` | `Solver` |
| `knudsen_codec(...)` | `Codec` |
| `isothermal_collider(unit, momentum_accommodation_coefficient=1.0, restitution=1.0, diffuse_sampling=DiffuseSampling.uniform)` | `Collider` |
| `volume_sink(unit, tolerance=0.0)` / `surface_sink(unit, tolerance=0.0)` / `tracing_sink(unit)` | `Sink` |
| `observer(interval, output_directory)` | `Observer` (CSV sampling) |

Enums: `DsmcKernelType` (`hard_sphere`, `variable_hard_sphere`, `variable_soft_sphere`)
and `DiffuseSampling` (`cosine_weighted`, `uniform`). Species lists are ordinary
Python lists of floats.

### Assembling and running

```python
system = atlas.build_system(
    fluid, universe, dt,
    solver=None, source=None, generator=None,
    colliders=[], sinks=[], codec=None, observer=None,
)
```

`fluid` and `universe` are **consumed** (moved into the System) — do not reuse the
handles afterwards. A source and generator are wired as a pair. `System` then
exposes:

- `update()` — advance one step; `save(directory)` — write a snapshot.
- `step`, `dt`, `particle_count`, `buffer_size`, `cell_count` — scalar read-backs.
- `positions()`, `velocities()` — `(N, 3)` float32 numpy arrays; `species()` — `(N,)` uint64.

The arrays are host copies of the live prefix, so they stay valid after the Fluid
handle has been consumed.

### Snapshots

`System.save(directory)` writes per-step binary snapshots. `load_fluid(path)` and
`load_universe(path)` rebuild device-resident state from a snapshot file.

## How the bindings are organized (for contributors)

`module.cpp` owns `NB_MODULE(atlas, ...)` and calls one `register_*` hook per
engine group; each hook lives in its own `bindings_<group>.cpp` and is declared in
[`register.h`](../../src/python/atlas/register.h). CMake globs the directory, so a
new file needs no build edit. The hooks run in dependency order — math and
geometry define the value types the later groups take as arguments, and the system
group ties everything together.

To add a binding, follow the existing files. A few rules that are easy to miss:

- **Core and Python API changes are one change.** Every new public Atlas Core
  builder, leaf, or assembly option must receive the matching Python factory or
  argument and be added to the API tables above in the same change. If the C++
  ownership model cannot be represented safely in Python, document that blocker
  here instead of exposing a borrowed object whose storage can expire.
- **Python handles carry borrowed-view owners.** Atlas Core keeps
  `TriangleMeshView` trivially copyable by borrowing buffers from a host-side
  `TriangleMesh`. The Python `Geometry`, `Unit`, `Source`, `Collider`, `Sink`,
  and `System` handles therefore propagate shared mesh owners alongside their
  Core values. `triangle_mesh(path)` users never need to retain a separate mesh
  object; the final `System` keeps every referenced mesh alive.

- **Prefer factory functions over the fluent builders.** Returning a built object
  from a lambda avoids the builders' reference-return lifetimes and keeps the
  Python surface flat.
- **Include the matching nanobind caster for every argument and return type**:
  `nanobind/stl/string.h`, `.../vector.h` (Python lists), `.../shared_ptr.h`,
  `.../unique_ptr.h`, `.../filesystem.h`, and `nanobind/ndarray.h` for numpy.
  A missing caster fails at build or import time.
- **Ownership maps to the std aliases.** `host_unique_ptr<T>` is
  `std::unique_ptr<T>` and `host_shared_ptr<T>` is `std::shared_ptr<T>`, so
  nanobind's native support applies. Owners built with `make_host_unique`
  (`Fluid`, `Universe`) are moved into `build_system`; policy objects built with
  `make_host_shared` (`Source`, `Generator`, `Solver`, `Codec`, `Observer`) are
  shared. Copyable value umbrellas (`Collider`, `Sink`, `Geometry`, `Material`)
  pass by value.
- **Do not return a consumed owner's object.** A raw pointer back to a moved-in
  `Fluid` resolves to the relinquished Python instance; expose the data you need
  (counts, arrays) on the `System` instead.
