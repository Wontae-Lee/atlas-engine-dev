# Python Bindings

The Python package exposes math, geometry, spatial queries, simulation objects,
sampling, and snapshot serialization. Public modules follow `include/atlas/`
and re-export the native nanobind classes and functions. Creation APIs are
PascalCase classes: `Fluid(...)`, `Sphere(...)`, `Molecule(...)`, and
`System(...)`. Classes are also exported directly from `atlas`. Free mathematical,
sampling, and serialization operations remain snake_case functions.

The root `src/python/atlas/module.cpp` is the single extension translation unit.
Registration headers mirror the corresponding C++ header paths and define inline
`register_<type>()` functions. The entry point owns `NB_MODULE(_core, m)`,
creates submodules, and arranges registration in dependency order.

## Building the module

The module is built when `ATLAS_PYTHON` is on. It needs the nanobind submodule
and a Python 3.8+ interpreter with development headers:

```bash
git submodule update --init --recursive external/nanobind
cmake -S . -B build/tbb -DATLAS_DEVICE_SYSTEM=TBB -DATLAS_PYTHON=ON -DATLAS_BENCHMARKS=OFF
cmake --build build/tbb --target atlas_python
```

The CMake target builds the private `_core` extension. A usable Python package
combines that extension with the modules under `src/python/atlas`; install the
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
(and `dist/cuda/` when both backends are built). The module is compiled with
nanobind's `STABLE_ABI`: on Python 3.12+ this yields one `abi3` wheel that serves
later versions, but on earlier interpreters the wheel is version-specific (e.g.
`cp311`), so build one per target Python. Install and use it:

```bash
pip install dist/tbb/atlas_engine-0.1.0-*.whl
python -c "import atlas; print(atlas.math.Bool3)"
```

The TBB wheel carries its own TBB runtime, so no system `libtbb` is required. A
CUDA wheel additionally needs a matching NVIDIA driver and a GPU at run time. To
build a wheel without Docker, install the front-end tools
(`pip install "scikit-build-core>=0.10" "nanobind>=2.0" build`) and run
`python -m build --wheel -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB`, but such a wheel
is not auditwheel-repaired and depends on the host's `libtbb`.

## GitHub CI and publication

`.github/workflows/tbb.yml` runs only for manual runs selecting `main`.
Its job guard excludes every other branch, including manual dispatches;
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
[`math/__init__.py`](../../src/python/atlas/math/__init__.py).

## Registration layout

- [`module.cpp`](../../src/python/atlas/module.cpp): the extension entry point,
  version, and native submodule registration order.
- [`math/vector/bool3.h`](../../src/python/atlas/math/vector/bool3.h):
  `atlas::python::math::register_bool3` registers Bool3. The remaining math
  types and functions have their own matching registration headers.
- [`math/__init__.py`](../../src/python/atlas/math/__init__.py): re-exports
  native types and functions under the public `atlas.math` path.
- [`CMakeLists.txt`](../../src/python/atlas/CMakeLists.txt): builds `module.cpp`
  into the private `_core` extension.

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
math, array transfers, spatial queries, and a deterministic step. See
[`tests/python/README.md`](../../tests/python/README.md).
