# Python binding tests

Build the TBB wheel, install it into the test environment, and run from the
repository root inside `python3 scripts/dev.py tbb` (use the CUDA shell for
the combined wheel):

```bash
ATLAS_WHEEL_BACKENDS=TBB python scripts/build_wheels.py
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

- `test_imports.py`: version, all public modules, exported native objects,
  PascalCase names, nested import identity, and the absence of low-level buffer exports.
- `test_engine_selection.py`: lazy native loading, installed engine discovery, TBB default
  preference, API/environment selection, rejection of invalid or missing engines,
  and prevention of changing the engine after native types have been imported.
  Each selection scenario runs in a fresh Python process. Cases requiring both
  extensions or a missing extension are skipped when the installed package does
  not provide that scenario.
- `test_construction.py`: state-owner, concrete leaf, policy, generator, and System
  construction through the public PascalCase API.
- `test_math.py`: constructors, writable components, arithmetic, checked
  indexing, explicit boolean reductions, and singular matrix handling.
- `test_fluid.py`, `test_universe.py`, and `test_numpy_transfer.py`: state
  lifecycles, range checks, compaction, dtype/shape contracts, direct transfer
  correctness, and owned snapshot lifetime.
- `test_geometry.py`, `test_material.py`, `test_source_generator.py`,
  `test_searcher.py`, `test_solver_codec.py`, and `test_boundaries.py`: focused
  module behavior without duplicating the C++ physics unit suite.
- `test_system.py`, `test_ownership.py`, and `test_integration.py`: core object
  hierarchy, individual phases, deterministic updates, consumed-object lifetime,
  mesh/policy retention, repeated collection, and the maintained Python example.
- `test_serialization.py`: temporary-file Fluid, Universe, and System snapshot
  round trips and snapshot independence.
- `test_errors.py`: controlled Python exceptions for invalid states, arrays, and
  configurations.
- `test_backend_parity.py`: deterministic TBB/CUDA comparison in isolated
  subprocesses. It skips when both extensions or a working CUDA runtime are not
  available.

Tests construct objects through public PascalCase classes, for example
`Fluid(...)`, `Sphere(...)`, `Molecule(...)`, and `System(...)`. Array-backed
population creation uses `Fluid.from_arrays(...)`. Abstract/umbrella classes
such as `Solver` are tested through their concrete subclasses.
