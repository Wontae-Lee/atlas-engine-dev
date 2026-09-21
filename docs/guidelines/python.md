# Python Bindings

The Python package exposes math, geometry, spatial queries, simulation objects,
sampling, and snapshot serialization. Public modules follow `include/atlas/`
and re-export the native nanobind classes and functions. Creation APIs are
PascalCase classes: `Fluid(...)`, `Sphere(...)`, `Molecule(...)`, and
`System(...)`. Classes are also exported directly from `atlas`. Free mathematical,
sampling, and serialization operations remain snake_case functions.

The root `bindings/python/atlas/module.cpp` is the single extension translation unit.
Registration headers mirror the corresponding C++ header paths and define inline
`register_<type>()` functions. The entry point builds as `_core_tbb` or
`_core_cuda`, creates submodules, and arranges registration in dependency order.
The `_core.py` facade connects the public imports to the selected extension.

## Default engine

Select the engine before importing any Atlas classes or computational submodules:

```python
import atlas

print(atlas.available_engines())
atlas.set_default_engine("cuda")
print(atlas.get_default_engine())

from atlas import Float3, Fluid, System, Universe
```

The accepted names are `tbb` and `cuda`. `available_engines()` lists installed
extensions, not GPU availability. With no explicit choice, TBB is preferred;
a CUDA-only installation defaults to CUDA. A missing requested extension raises
an error instead of selecting another engine.

`ATLAS_DEFAULT_ENGINE=cuda` (or `tbb`) sets the initial default for a process.
`set_default_engine()` can override it until the native extension is loaded.
Importing `atlas` and querying the engine do not load either native extension.
The first class or computational submodule import fixes the engine for that
process. Re-selecting the same engine is allowed; selecting the other afterward
raises `RuntimeError`. Start a new process to use another engine. Objects and
class references already imported are never migrated to another backend.

Both engines expose the same PascalCase API and NumPy host-array interface.
CUDA executes the engine's device kernels on the GPU; TBB executes the CPU
implementation. Parallel floating-point results need not be bit-for-bit equal.

## Docker runtime

The `tbb` and `cuda` Docker targets install Python, NumPy, and the Atlas wheel
in `/opt/venv`; `python` and `pip` use that environment automatically. Each
runtime image includes only its selected native extension and sets
`ATLAS_DEFAULT_ENGINE` accordingly. Unlike the combined CUDA wheel built by
`scripts/build_wheels.sh`, the CUDA runtime image does not include a TBB engine.

```bash
git submodule update --init --recursive
docker build --target tbb -t atlas:tbb .
docker run --rm atlas:tbb python /opt/atlas/examples/python/dsmc_dense_cell.py
docker run --rm -v "$PWD":/workspace atlas:tbb python simulation.py
```

For CUDA, build with `--target cuda`, use the resulting image, and add
`--gpus all` to `docker run`. See [docker.md](docker.md) for host GPU setup,
Ubuntu versions, and development targets. The `wheel` target below is a
packaging toolchain and does not have Atlas preinstalled.

## Building the module

The module is built when `ATLAS_PYTHON` is on. It needs the nanobind submodule
and a Python 3.8+ interpreter with development headers:

```bash
git submodule update --init --recursive external/nanobind
cmake -S . -B build/tbb -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_PYTHON=ON -DATLAS_BENCHMARKS=OFF
cmake --build build/tbb --target atlas_python
```

The CMake target builds the private `_core_tbb` or `_core_cuda` extension. A usable Python package
combines that extension with the modules under `bindings/python/atlas`; install the
project (or a wheel as described below) before importing it:

```bash
python -m pip install -e .
python -c "import atlas; print(atlas.math.Bool3)"
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
(and `dist/cuda/` when CUDA is requested). The CUDA wheel includes both native
extensions, so either engine can be selected after installing that one wheel.
The TBB wheel contains only `_core_tbb` and has no CUDA runtime dependency.
Both extensions in a CUDA wheel are built with the same Python interpreter.
The module is compiled with
nanobind's `STABLE_ABI`: on Python 3.12+ this yields one `abi3` wheel that serves
later versions, but on earlier interpreters the wheel is version-specific (e.g.
`cp311`), so build one per target Python. Install and use it:

```bash
pip install dist/tbb/atlas_engine-0.1.0-*.whl
python -c "import atlas; print(atlas.math.Bool3)"
```

The TBB wheel carries its own TBB runtime, so no system `libtbb` is required.
Selecting CUDA additionally needs a matching NVIDIA driver and a GPU at run time;
selecting TBB from the combined wheel does not load the CUDA extension. To
build a wheel without Docker, install the front-end tools
(`pip install "scikit-build-core>=0.10" "nanobind>=2.0" build`) and run
`python -m build --wheel -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB`, but such a wheel
is not auditwheel-repaired and depends on the host's `libtbb`.

`scripts/build_wheels.sh` builds TBB first even when only CUDA output is
requested, then passes its raw extension to the CUDA build through
`ATLAS_PYTHON_TBB_EXTENSION`. Direct CMake/scikit-build builds produce only the
selected engine unless this path is supplied. Do not install separate TBB and
CUDA wheels on top of each other to combine them: they share a distribution
name. Install the combined CUDA wheel instead.

## GitHub CI and publication

The separate `.github/workflows/python.yml` checks a TBB source distribution
and installed wheel on Ubuntu 22.04/Python 3.10 and Ubuntu 24.04/Python 3.12.
It checks metadata and engine selection, runs the full Python test suite,
and executes the DSMC example. C++ tests run in `tbb.yml`.

Both workflows are manual and restricted to `main`. The independent
`publish-python.yml` builds manylinux TBB release wheels; PyPI upload requires
the publish input. See [releases.md](releases.md) for artifacts, version
metadata, publication, and citation maintenance.

## Quick start

```python
from atlas.math import Bool3, Float3, Float3x3, Int3, Quaternion, dot

mask = Bool3(True, False, True)
assert mask.any()
assert not mask.all()

velocity = Float3(1.0, 2.0, 3.0)
assert dot(velocity, velocity) == 14.0
assert Int3(1, 2, 3).z == 3
assert Float3x3(1.0) * velocity == velocity
assert Quaternion().rotate(velocity) == velocity
```

## Math API

- `Bool3`: default, copy, and component constructors; writable `x/y/z`;
  `all/any/none`; component-wise `&`, `|`, and `~`.
  Implicit truth conversion raises `TypeError`; choose a reduction explicitly.
- `Int3`: constructors, components, checked indexing (including negative
  indices), arithmetic, comparisons, extrema, and zeroing.
- `Float3`: constructors, components, checked indexing, scalar/vector
  arithmetic, comparisons, dot/cross products, length, normalization,
  reflection, projection, and tangent queries.
- `Float3x3`: constructors, nine writable components, flat or row/column
  indexing, arithmetic, determinant, transpose, inverse, and linear solves.
  `try_inverse()` and `solve()` return `None` on failure.
- `Quaternion`: component, axis-angle, Euler, and matrix construction;
  writable components, indexing, arithmetic, normalization, interpolation,
  rotation, and matrix conversion.

Free math helpers include `dot`, `cross`, `clamp`, `normalized_or`,
`orthonormal_basis`, `spherical_direction`, matrix operations, and
`solve_quadratic`. Constants include `pi`, `SQRT_TWO`,
`boltzmann_constant`, `gravity`, `eps`, `tol`, `far`, and `inf`.
The explicit export list lives in
[`math/__init__.py`](../../bindings/python/atlas/math/__init__.py).

## Registration layout

- [`module.cpp`](../../bindings/python/atlas/module.cpp): the extension entry point,
  version, and native submodule registration order.
- [`math/vector/bool3.h`](../../bindings/python/atlas/math/vector/bool3.h):
  `atlas::python::math::register_bool3` registers Bool3. The remaining math
  types and functions have their own matching registration headers.
- [`math/__init__.py`](../../bindings/python/atlas/math/__init__.py): re-exports
  native types and functions under the public `atlas.math` path.
- [`CMakeLists.txt`](../../bindings/python/atlas/CMakeLists.txt): builds `module.cpp`
  into the selected native extension and assigns an engine-specific nanobind
  domain so native type registries cannot be confused.
- [`_engine.py`](../../bindings/python/atlas/_engine.py): validates the selected engine
  and loads it once. The root package lazily resolves PascalCase exports.

There is no separate `bind/` tree, generic registration wrapper, or per-type
registration class. Add registration headers under the matching C++ paths,
call them from `module.cpp`, and add public names to the module's `__init__.py`.
Include the matching nanobind caster
for every STL argument or return type. Native source/header files are excluded
from wheels; they remain in source distributions.

The wheel smoke command checks math imports and operations. The simulation
example uses the PascalCase classes exported from `atlas`.

## Registered modules

| Public module | Registered API |
|---|---|
| `math` | Vectors, matrices, quaternions, constants, and free math functions |
| `spatial` | Ray, hit results, AABB queries/transforms, BVH, LBVH, and SAHBVH |
| `geometry` | Geometry queries and primitive/mesh classes |
| `sync`, `unit` | Pose transforms and moving geometry |
| `material` | Material types, species classes, and MaterialDictionary |
| `fluid`, `universe` | State owners, array access, creation, and snapshot loading |
| `source`, `generator` | Emission sources and velocity/species generators |
| `solver`, `codec` | DSMC kernel/solver and Knudsen codec |
| `collider`, `sink` | Isothermal reflection and volume/surface/tracing removal |
| `observer`, `system` | Sampling counters, output, assembly, and stepping |
| `searcher` | Spatial hashing and host copies of search results |
| `random`, `sampling` | Random engines, distributions, seeds, and sampling helpers |
| `serialization` | Binary snapshot reading, writing, and state restoration |

Nested Python import paths also expose `math.vector`, `math.matrix`,
`spatial.bounding_volume_hierarchy`, `solver.dsmc`, and `solver.dsmc.kernel`.
They reference the same registered types as their parent modules.

The mirror describes registration file locations, not complete coverage of
every C++ declaration. Internal template infrastructure (`core`, `buffer`,
`container`, `memory`, `parallel`, and `scan`), device views, logging, and
individual low-level kernel implementations have no separate Python bindings.
Concrete leaf classes such as `Sphere`, `Molecule`, `VolumeSource`, and
`UniformGenerator` are Python subclasses of their registered umbrella type.
Their constructors are bound in the corresponding C++ registration header with
nanobind's `nb::new_`. Nanobind retains the native owner and gives the result the
requested Python subtype, so `type(Sphere(...)) is Sphere` and
`isinstance(Sphere(...), Geometry)` both hold. Methods operate on the same native
storage; there is no Python attribute-forwarding data wrapper.

Shared native adapters live in `_detail/`: array/state conversion, BVH and
search helpers, and handles that retain mesh and policy owners. These preserve
the lifetime of C++ objects referenced by Geometry, Unit, Source, Collider,
Sink, and System. `System(...)` consumes its Fluid and Universe arguments;
read or modify their state through the resulting System afterward.

`System` uses an in-place `__init__` binding so move-only arguments are consumed
once. Other owner constructors use `nb::new_` to retain their C++ builder's
ownership semantics. Alternative state creation is available as
`Fluid.from_arrays()`, `Fluid.load()`, `Universe.from_geometry()`, and
`Universe.load()`. `TriangleMesh` overloads its constructor for an OBJ path,
triangle list, or vertex/index arrays; `Plane` accepts either normal/offset or
point/normal arguments.

```python
from atlas import Float3, Fluid, Sphere, System, Unit, Universe, VolumeSink

shape = Sphere(Float3(0.5), 0.1)
boundary = VolumeSink(Unit(shape))
simulation = System(
    Fluid(buffer_size=16),
    Universe(Float3(0), Float3(1), cell_size=0.5),
    dt=0.01,
    sinks=[boundary],
)
```

Run `python -X faulthandler -m unittest discover -s tests/python -v` against an
installed wheel to check constructor identity, module exports, native ownership,
math, array transfers, spatial queries, and a deterministic step. Set
`ATLAS_DEFAULT_ENGINE=tbb` and `ATLAS_DEFAULT_ENGINE=cuda` in separate runs
to check both backends from the combined wheel. See
[`tests/python/README.md`](../../tests/python/README.md).
