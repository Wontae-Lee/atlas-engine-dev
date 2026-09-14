# Python Bindings

Atlas ships an optional [nanobind](https://nanobind.readthedocs.io) extension
module, `atlas`, that exposes the C++ simulation builders, state access, pipeline
phases, and numerical queries so a DSMC simulation can be assembled, modified,
stepped, and read back from Python. The engine itself is unchanged;
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

Array read-back returns NumPy arrays, so `numpy` is a declared runtime dependency
(`pyproject.toml`) and is installed with the wheel. State arrays, observer counters,
searcher arrays, and decoded snapshot arrays are owned host copies. Updating one
does not update the simulation; use the state setters to upload changes.

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
pip install dist/tbb/atlas_engine-0.1.0-*.whl
python -c "import atlas; print(atlas.build_system)"
```

The TBB wheel carries its own TBB runtime, so no system `libtbb` is required. A
CUDA wheel additionally needs a matching NVIDIA driver and a GPU at run time. To
build a wheel without Docker, install the front-end tools
(`pip install "scikit-build-core>=0.10" "nanobind>=2.0" build`) and run
`python -m build --wheel -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB`, but such a wheel
is not auditwheel-repaired and depends on the host's `libtbb`.

## GitHub CI and publication

`.github/workflows/tbb.yml` runs on pushes to `main` and manual runs selecting
`main`. Its job guard excludes every other branch, including manual dispatches;
there is no pull-request trigger. It runs the aggregate C++ GoogleTest suite,
builds a source archive, builds and installs a wheel from that archive, checks
metadata/version consistency, and executes `examples/python/dsmc_dense_cell.py`.
The `tbb-linux-cp311` artifact is a CI build, not an auditwheel-repaired release.

`.github/workflows/publish-python.yml` is manual and also restricted to `main`.
Its default is to build artifacts only. Selecting the `publish` input enables
PyPI Trusted Publishing after source and wheel checks pass; register workflow
`publish-python.yml` and environment `pypi` in PyPI and create that GitHub environment.
Tag pushes no longer publish Python packages. GitHub releases used by Zenodo are
independent of this workflow.

The release workflow builds TBB manylinux x86_64 wheels for CPython 3.9–3.13 from
the source archive, so missing submodule files are caught before publication.
`sdist.include` explicitly includes those dependencies; Git internals and local
build outputs are excluded. Source builds need system TBB and network access for
Abseil. The wheel includes the project and vendored dependency license files.
`SKBUILD_PROJECT_VERSION` supplies the extension's version during packaging;
update `project.version` and citation metadata together for each release.

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

C++ simulation builders are exposed as **factory functions** that run the builder
and return a ready object; value primitives and standalone acceleration builders
are classes with constructors. Factory names are lower-case. The returned objects
also expose queries and operations described below.

### Math and geometry

| Call | Returns |
|---|---|
| `Float3(x, y, z)` / `Quaternion(w, x, y, z)` / `Quaternion(axis, radians)` | value types |
| `Bool3(x, y, z)` / `Int3(x, y, z)` / `Float3x3(...)` | mask, integer vector, and matrix |
| `Ray(origin, direction)` / `AABB(lower, upper)` | spatial value types |
| `sphere(center, radius)` | `Geometry` |
| `plane(normal, offset)` | `Geometry` |
| `plane_from_point(point, normal)` | `Geometry` |
| `box(lower, upper)` | `Geometry` |
| `cylinder(center, radius, height, open=False)` | `Geometry` |
| `circle(center, normal, radius)` | `Geometry` |
| `square(center, normal, side_length)` | `Geometry` |
| `triangle(a, b, c, normal=None)` | `Geometry` |
| `polygonal_prism(center, side_count, radius, height)` | `Geometry` |
| `triangle_mesh(path, verbose=False)` | `Geometry` loaded from a Wavefront OBJ file |
| `triangle_mesh_from_triangles(triangles)` | `Geometry` from a list of triangle geometries |
| `triangle_mesh_from_arrays(vertices, indices, normals=None)` | `Geometry` from `(N, 3)` float32 vertices, `(M, 3)` int64 indices, and optional `(M, 3)` float32 face normals |

Vectors expose arithmetic, component access, and comparisons. Equality and
inequality return `bool`; ordering comparisons return `Bool3`, reduced with
`all()`, `any()`, or `none()` explicitly. `Float3` also exposes
dot/cross products, normalization, projection, reflection, and tangent vectors.
`Float3x3` supports flat or `(row, column)` indexing, matrix/vector multiplication,
transpose, determinant, inverse, and linear solves. `try_inverse()` and `solve(b)`
return `None` on failure. Quaternions support Euler/matrix/axis-angle construction,
rotation, normalization, conjugation, inversion, and lerp/nlerp/slerp.

The corresponding free math helpers are exposed under `atlas`, including
`dot`, `cross`, `clamp`, `normalized_or`, `orthonormal_basis`,
`spherical_direction`, matrix operations, and `solve_quadratic` (a root pair or
`None`). Constants include `pi`, `boltzmann_constant`, `gravity`, `eps`, `tol`,
`far`, `inf`, and `SQRT_TWO`.

`Geometry` exposes `type`, `closest_point`, `closest_normal`, `signed_distance`,
`is_inside`, `is_on_surface`, `centroid`, `bound`, `is_valid`, and `trace(ray)`.
Triangle meshes additionally support `winding_number(point)`. `trace` returns
`HitSurface` (`is_intersecting`, `distance`, `point`, `normal`); AABB tracing
returns `HitAABB` (`is_intersecting`, `enter`, `exit`). AABBs expose dimensions,
overlap/containment checks, corners, merging, expansion, clamping, and distances.
`transform_aabb` accepts either a `Sync` or a Python point-transform callable.

`LBVH(triangles=[], morton_bits=10)` and
`SAHBVH(triangles=[], leaf_size=32, bin_count=100)` build standalone acceleration
structures from lists of triangle geometries. Both expose `build(triangles)`,
`reset()`, `root`, and their tuning properties/setters. `nodes()`, `indices()`,
`bounds()`, and `centroids()` return independent Python lists; `BVHNode` exposes
its bound, tree links, primitive range, leaf flag, and solid-angle moments.
Changing tuning parameters affects the next build. Raw BVH device views remain
C++ interfaces; use `Geometry` for geometric queries from Python.

### Random numbers and sampling

`DefaultRandomEngine(seed=1)` exposes the engine's C++ sequence through calls,
`seed(value)`, and `discard(count)`, with `min()`, `max()`, and the engine
constants. `UniformRealDistribution(min_value=0, max_value=1)` samples float
values with `distribution(engine)`; `UniformRealDistributionDouble` exposes
the double instantiation.

The sampling functions include `generate_standard_normal_pair` (a pair),
`generate_standard_normal`, `sample_uniform_vector`, `sample_normal_vector`,
`build_orthonormal_basis` (tangent/bitangent pair),
`sample_uniform_hemisphere`, `sample_cosine_hemisphere`,
`sample_random_unit_vector`, `sample_directional_unit_vector`,
`sample_axis_count`, `shuffle_key`, and hashed interval/index draws.
`sample_weighted_index(weights, engine)` and
`sample_weighted_choice(weights, values, engine)` take Python lists in place
of C++ pointer/count arguments. They retain the core contract of normalized
non-negative weights; the latter truncates the chosen non-negative float value
to an integer. `DEFAULT_UNSIGNED_INT_SEED` is also exported.

### Transform, material, and the two state owners

| Call | Returns |
|---|---|
| `sync(translation, orientation=Quaternion())` | `Sync` (a rigid pose) |
| `unit(geometry, sync=None, velocity=None, angular_velocity=None, acceleration=None, angular_acceleration=None)` | `Unit` (a placed body) |
| `molecule / atom / ion / neutron(mass, translational_energy, rotational_energy, vibrational_energy, reference_diameter, reference_temperature, viscosity_index, scattering_parameter)` | `Material` |
| `solid(mass)` | `Material` |
| `material_dictionary([Material, ...])` | `MaterialDictionary` |
| `fluid(buffer_size, particle_count=0, statistical_weight=1.0, materials=None)` | `Fluid` |
| `fluid_from_arrays(positions, velocities, statistical_weight=1.0, materials=None, *, species=None, buffer_size=None)` | `Fluid` seeded from `(N, 3)` arrays, with optional `(N,)` uint64 species and spare capacity |
| `universe(lower, upper, cell_size)` | `Universe` |
| `universe_from_geometry(geometry, cell_size)` | `Universe` |

`Sync` exposes translation/orientation setters, cached rotation matrices, and
point/direction/ray transforms between local and world coordinates. `Unit`
exposes geometry, pose, kinematic readbacks, `update`, `move`, `rotate`,
`set_geometry`, `set_sync`, `trace`, `surface_velocity`, and `world_bound`.
Returned poses and units are value copies: edit a pose and pass it to
`unit.set_sync(pose)` to replace the stored pose. Mesh owners accompany every
copied geometry or unit.
Assign `pose.translation` or `pose.orientation` to update a `Sync`; modifying
a component on a returned value copy does not update the stored pose.

`Material.type` identifies its species kind. `MaterialDictionary` supports
`len`, indexing (including negative indices), entry replacement, `materials()`
as a host list, and `set_materials(list)`. Keep particle species ids valid when
changing the table. Maxwell–Boltzmann generators cache their masses when built;
rebuild a generator to use a changed material table.

`Fluid.particle_count` is writable and cannot exceed `buffer_size`;
`set_particle_count(count)` is equivalent. `statistical_weight` and `materials`
are read-only properties. `Universe` exposes `cell_count`, `lower_corner`,
`upper_corner`, `grid_size`, `cell_size`, `cell_volume`, and `inverse_cell_size`.

### Particle and grid state

Both `Fluid` and `Universe` expose `state(name)`, `set_state(name, values,
offset=0)`, `has_state(name)`, `remove_state(name)`, and `reset_state(name)`.
Unknown names raise `KeyError`; reading a known but unattached state returns
`None`. `set_state` creates an absent field at the owner's full capacity and
writes the supplied rows starting at `offset`, preserving the remaining rows.
Writes beyond capacity raise `ValueError`. `remove_state` returns whether a
field was present; `reset_state` zeroes an attached field and raises `KeyError`
when it is absent.

| Owner | State names | Array dtype / shape |
|---|---|---|
| Fluid | `position`, `velocity` | float32 `(N, 3)` |
| Fluid | `species` | uint64 `(N,)` |
| Fluid | `temperature`, `translational_energy`, `rotational_energy`, `vibrational_energy` | float32 `(N,)` |
| Universe | `bulk_velocity`, `field_force`, `gravity` | float32 `(cell_count, 3)` |
| Universe | `temperature`, `max_relative_speed`, `max_sigma_g`, `thermal_energy`, `number_particle`, `knudsen_number` | float32 `(cell_count,)` |
| Universe | `collision_count`, `allocated_solver` | int32 `(cell_count,)` |

Fluid reads return the live prefix (`N = particle_count`); pass `full=True` to
`state` to inspect all `buffer_size` slots. `positions()`, `velocities()`, and
`species()` are convenient live-prefix readbacks. `active(full=False)` and
`set_active(values, offset=0)` access the int32 survivor flags separately from
the state store; flags must be 0 or 1. `compact()` keeps flagged live particles
and compacts every attached column, updating `particle_count`.

`fluid_from_arrays` initializes species to zero when omitted. Its capacity is
`buffer_size` when supplied, otherwise `max(N, 1)`. Every column retains that
capacity even when the initial arrays are empty, allowing subsequent emission
and snapshot restoration. For manual initialization through `set_state`, set
`particle_count` after filling the desired particle rows.

Setting a field only stores data: fields reserved in the C++ engine (such as
gravity) retain the same pipeline behavior in Python. Removing required fields
has the same consequences as removing them in C++; `System.initialize_states()`
can restore the universe fields required by its solvers.

### Emitter, solver, and boundaries

| Call | Returns |
|---|---|
| `volume_source(unit, spacing, tolerance=0.0)` / `surface_source(...)` | `Source` |
| `maxwell_boltzmann_generator(species_ratios, species_numbers, materials=None, temperature=273.15, bulk_velocity=Float3(), seed=0, species_mass=None)` | `Generator`; explicit masses take precedence over the dictionary |
| `uniform_generator(species_ratios, species_numbers, temperature, min_value, max_value, bulk_velocity, seed)` | `Generator` |
| `jittering_generator(species_ratios, species_numbers, temperature, base_value, jitter_radius, bulk_velocity, seed)` | `Generator` |
| `maxwell_sigma_generator(species_ratios, species_numbers, temperature, sigma, bulk_velocity, seed)` | `Generator` |
| `dsmc_solver(kernel_type=DsmcKernelType.variable_hard_sphere, majorant_sample_pairs=8, majorant_exhaustive_limit=5)` | `Solver` |
| `knudsen_codec(...)` | `Codec` |
| `isothermal_collider(unit, momentum_accommodation_coefficient=1.0, restitution=1.0, diffuse_sampling=DiffuseSampling.uniform)` | `Collider` |
| `volume_sink(unit, tolerance=0.0)` / `surface_sink(unit, tolerance=0.0)` / `tracing_sink(unit)` | `Sink` |
| `observer(interval, output_directory)` | `Observer` (CSV sampling) |

Enums: `DsmcKernelType` (`hard_sphere`, `variable_hard_sphere`, `variable_soft_sphere`),
`DiffuseSampling` (`cosine_weighted`, `uniform`), and the `GeometryType`,
`MaterialType`, `SourceType`, `GeneratorType`, `ColliderType`, `SinkType`,
`SolverType`, and `CodecType` tags. Species lists are ordinary Python lists of floats.

`Source` exposes `unit`, `cached_count`, `advance(dt)`, and `spawn(fluid, offset=0)`.
`Generator` exposes `temperature`, writable `bulk_velocity`,
`set_bulk_velocity(value)`, and `generate(fluid, offset, count)`. Standalone
spawn/generate operations return the actual number of rows written and leave
`particle_count` unchanged; spawning writes only positions, and generation
writes velocity/species. `System.emit()` performs the complete paired operation.

`Collider` exposes its unit, bound, accommodation coefficient, `advance`, `trace`,
`reflect`, and `collide(hit, position, velocity, dt)`, which returns the new
position/velocity pair. `Sink` exposes its unit, `advance`, and
`despawn(position, velocity, dt)`. The unit getters return copies.

`dsmc_solver` returns a `DsmcSolver` (also a `Solver`) with read-only kernel and
majorant settings. `DsmcKernel(kernel_type)` exposes `cross_section`,
`sigma_g(lhs, rhs, relative_speed_squared)`, and `scatter`, returning the pair
of scattered velocities. `scatter(..., seed=0)` starts a fresh random stream;
`scatter(..., engine=engine)` advances a supplied `DefaultRandomEngine`.
`Codec` exposes `split_count`, `knudsen_number`,
`solver_index`, and `allocate(universe)` over already attached states.

### Assembling and running

```python
system = atlas.build_system(
    fluid, universe, dt,
    solver=None, source=None, generator=None,
    colliders=[], sinks=[], codec=None, observer=None,
    solvers=[], emitters=[],
)
```

`fluid` and `universe` are **consumed** (moved into the System) — do not reuse the
handles afterwards. A source and generator must be supplied together; a partial
pair raises `ValueError`. The keyword-only `solvers` list and `emitters` list of
`(source, generator)` tuples support repeated C++ builder additions. The singular
`solver` and `source`/`generator` arguments, when supplied, precede the respective
list entries. Solver indices used by a codec or `allocated_solver` state refer
to that final order. `System` then exposes:

- `update()` — advance one step; `save(directory)` — write a snapshot.
- `step`, `dt`, `buffer_size`, `cell_count`, `source_count`, `solver_count`,
  `collider_count`, `sink_count` — scalar read-backs; `particle_count` is writable.
- `positions()`, `velocities()` — `(N, 3)` float32 numpy arrays; `species()` — `(N,)` uint64.
- `fluid_state(name, full=False)`, `set_fluid_state(name, values, offset=0)`,
  `has_fluid_state`, `remove_fluid_state`, `reset_fluid_state` — the owned fluid's fields.
- `universe_state(name)`, `set_universe_state(name, values, offset=0)`,
  `has_universe_state`, `remove_universe_state`, `reset_universe_state` — the owned grid's fields.
- `active(full=False)`, `set_active(values, offset=0)`, `compact()` — survivor handling.
- The same fluid weight/material and universe grid geometry properties listed above.
- `solvers`, `codec`, `observer` — shared policy handles.
- `emit()`, `search()`, `allocate()`, `solve()`, `advect()`, `remove()` and
  `initialize_states()` — individual C++ phases. They do not increment `step`
  or notify the observer; `update()` performs the complete step.
- `observe()` — invoke the attached observer at the current step, respecting its interval.

The arrays are host copies of the live prefix, so they stay valid after the Fluid
handle has been consumed.

After editing particle positions, counts, or ordering, call `search()` before a
manual `solve()` so its cell ranges describe the current particles.

### Observation and spatial searching

`Observer` exposes `interval`, `output_directory`, `species_count`,
`reset_counters()`, `resize_counters(source_count, sink_count, species_count)`,
and `observe(fluid, universe, step)` or `observe(system)`. `spawned()` and
`despawned()` return owned int32 matrices shaped `(source_count, species_count)`
and `(sink_count, species_count)`. `System` initially sizes these counters;
manual resizing must match the system's source/sink/species configuration.

`spatial_hashing_searcher(universe)` or
`spatial_hashing_searcher(lower_corner, cell_size, grid_size)` creates a standalone
`SpatialHashingSearcher`. `classify(fluid, universe=None)` bins the live particles;
the optional universe must match the grid and have `number_particle` attached
to receive per-cell counts. `cell_key()`, `indices()`, `cell_start()`, and
`cell_end()` return owned arrays. Grid properties, `particle_count`, `reset()`,
`cell_for(position)`, `contains_cell(cell)`, and `linear_key(cell)` are exposed.
`solve(solver, fluid, universe, index, dt)` passes the internal searcher view to
a solver without exposing its device pointers. The universe must have the
solver's required fields, and the grid and particle count must match the last
classification. Reclassify after changing positions, ordering, or population.

### Snapshots

`System.save(directory)` writes per-step binary snapshots. `load_fluid(path)` and
`load_universe(path)` rebuild device-resident state from a snapshot file.
`restore_fluid(path)` and `restore_universe(path)` expose the C++ names for the
same restoration operation. `save_fluid_binary(fluid_or_system, path)` and
`save_universe_binary(universe_or_system, path)` save individual files.

`load_fluid_binary(path)` and `load_universe_binary(path)` decode to Python
dictionaries with metadata and owned host arrays, without constructing a
device-resident fluid or universe. Missing fields are `None`; fluid arrays
cover the full buffer capacity, and the dictionary's `particle_count` identifies
the live prefix. The fluid dictionary also contains copied `Material` objects.

### C++ coverage boundary

The bindings cover simulation assembly, existing solver/generator/boundary
choices, built-in state columns, observation, snapshots, and host numerical and
geometry queries. This is functional coverage of those workflows, not a binding
of every C++ declaration.

C++ extension machinery remains C++: custom solver/state/variant leaf classes,
`TypeStore<T>`, general `Container<T, N>`, raw `HostBuffer`/`DeviceBuffer` storage,
device pointer views, and user-written parallel kernels, atomics, sorts, and
scans. Python works on owned host arrays instead of borrowing device addresses.
Methods made public only to satisfy nvcc's extended-lambda restriction
(`System.mark_survivors`, `record_spawned`, and the searcher's internal passes)
are intentionally accessed through their complete host operations.

## How the bindings are organized (for contributors)

`module.cpp` owns `NB_MODULE(atlas, ...)` and calls one `register_*` hook per
engine group; each hook lives in its own `bindings_<group>.cpp` and is declared in
[`register.h`](../../src/python/atlas/register.h). CMake globs the directory, so a
new file needs no build edit. The hooks run in dependency order — math and
geometry define the value types the later groups take as arguments, and the system
group ties everything together. Sampling follows math, BVH follows geometry,
and standalone searcher/serialization hooks follow the system registration.

CUDA builds compile every binding translation unit with nvcc: the inline
`DeviceBuffer` constructors, copies, and state operations instantiate Thrust's
CUDA implementation. Compiling only `module.cpp` with nvcc is insufficient.
TBB builds continue to use the host compiler for the bindings.

`array_utils.h` handles owned NumPy copies and bounded uploads. `state_access.h`
maps Python state names to the existing C++ field types; it creates optional
fields at their owner's full capacity and never exposes borrowed state pointers.

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
- **Release Python-backed policy handles on the calling thread.** CUDA
  `HostBuffer` destruction can run on TBB workers. Nanobind's shared-pointer
  deleter acquires the Python GIL, so destroying its last reference on a worker
  while the calling thread holds the GIL and waits for TBB would deadlock.
  `PySystem` retains the same solver, source, and generator control blocks in a
  `std::vector`, declared before the Core `System` so it is destroyed afterward
  on the calling thread. `build_system` retains these handles before constructing
  the builder, which also protects validation failures and temporary buffers.

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
