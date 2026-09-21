# Python binding tests

Build the TBB wheel, install it into the test environment, and run from the
repository root:

```bash
ATLAS_WHEEL_BACKENDS=TBB bash scripts/build_wheels.sh
python -m pip install --force-reinstall dist/tbb/atlas_engine-*.whl
ATLAS_DEFAULT_ENGINE=tbb PYTHONDONTWRITEBYTECODE=1 python -X faulthandler -m unittest discover -s tests/python -v
```

To exercise the CUDA backend, build and install the CUDA wheel and run the same
suite with `ATLAS_DEFAULT_ENGINE=cuda`. The runtime tests allocate device memory
and execute the simulation, so this run requires a working CUDA driver and GPU.
Engine-selection tests only import native types and construct host math values;
they do not allocate GPU state.

The tests use `unittest` and NumPy, a runtime dependency of the wheel. They
import the installed `atlas` package without modifying `sys.path` or loading an
extension from an old build directory.

- `test_modules.py`: version, all 20 public modules, exported native objects,
  PascalCase names of registered classes, and nested import identity.
- `test_engine.py`: lazy native loading, installed engine discovery, TBB default
  preference, API/environment selection, rejection of invalid or missing engines,
  and prevention of changing the engine after native types have been imported.
  Each selection scenario runs in a fresh Python process. Cases requiring both
  extensions or a missing extension are skipped when the installed package does
  not provide that scenario.
- `test_constructors.py`: concrete PascalCase classes, exact constructor result
  types, umbrella inheritance, all primitive/material/generator/policy classes,
  mesh constructor overloads, and emitter ownership after Python handles expire.
- `test_math.py`: constructors, writable components, arithmetic, checked
  indexing, explicit boolean reductions, and singular matrix handling.
- `test_runtime.py`: geometry/pose queries, mesh ownership, BVH construction,
  array transfers, spatial hashing, sources/generators, material tables,
  boundaries, seeded sampling, binary snapshots, and a deterministic System step.
- `test_cuda_parity.py`: mesh collider steps, mesh sink removal and compaction,
  device material-table writes, and momentum/energy conservation for all three
  DSMC collision models. Run the same tests with each engine selected.

Tests construct objects through public PascalCase classes, for example
`Fluid(...)`, `Sphere(...)`, `Molecule(...)`, and `System(...)`. Array-backed
population creation uses `Fluid.from_arrays(...)`. Abstract/umbrella classes
such as `Solver` are tested through their concrete subclasses.
