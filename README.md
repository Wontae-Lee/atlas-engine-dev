<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine

[![DOI](https://zenodo.org/badge/1021472310.svg)](https://doi.org/10.5281/zenodo.22752178)
[![Core CI](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-core.yml)
[![Python CI](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-python.yml)
[![PyPI](https://img.shields.io/pypi/v/atlas-engine.svg)](https://pypi.org/project/atlas-engine/)

Atlas is a C++20 particle simulation engine for rarefied-gas flows using direct
simulation Monte Carlo (DSMC). It runs on CPUs through TBB or NVIDIA GPUs
through CUDA. The Python package exposes the same core simulation objects, while
the native Interactive subsystem provides JSONL control and OpenGL rendering.

Atlas includes:

- hard-sphere, variable hard sphere (VHS), and variable soft sphere (VSS)
  collision models;
- analytic shapes and OBJ triangle meshes;
- configurable particle sources, generators, colliders, and sinks;
- per-particle and per-cell state with protobuf snapshots;
- owned NumPy snapshots through the Python binding;
- native headless simulation control and real-time particle rendering.

## Choose a workflow

| Goal | Start here |
|---|---|
| Run Atlas in a container | [Docker quick start](#docker) |
| Install and use the Python package | [Python](#python) |
| Build a native C++ simulation | [C++](#c) |
| Run JSONL control or the OpenGL window | [Interactive](#interactive) |
| Compare the maintained cylinder case across all frontends | [Examples](examples/README.md) |

Clone the repository with its build dependencies:

```bash
git clone --recurse-submodules https://github.com/Wontae-Lee/atlas-engine-dev.git
cd atlas-engine-dev
```

If the repository is already cloned, initialize dependencies with
`git submodule update --init --recursive`.

## Docker

Build and run the bundled Python cylinder case on TBB:

```bash
docker build --target tbb -t atlas:tbb .
docker run --rm atlas:tbb \
    python /opt/atlas/examples/python/main.py cylinder 40
```

For CUDA, the host needs an NVIDIA driver and NVIDIA Container Toolkit:

```bash
docker build --target cuda -t atlas:cuda .
docker run --rm --gpus all atlas:cuda \
    python /opt/atlas/examples/python/main.py cylinder 40
```

The runtime images also contain the native Interactive executables. Published
image tags, Ubuntu variants, GPU/display access, and development images are
covered by the [Docker guide](docs/operations/docker.md).

## Python

Atlas is distributed as `atlas-engine` and imported as `atlas`. A local TBB
source installation on Ubuntu needs a C++ toolchain, CMake, Ninja, TBB, and
Python development headers:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git libtbb-dev python3-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install . -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB
```

Select an installed engine before importing native classes, then construct and
advance the core objects directly:

```python
import atlas
import numpy as np

atlas.set_default_engine("tbb")
from atlas import Float3, Fluid, System, Universe

fluid = Fluid.from_arrays(
    np.array([[0.25, 0.5, 0.5]], dtype=np.float32),
    np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
)
universe = Universe(Float3(0), Float3(1), cell_size=0.5)
universe.set_state(
    "number_particle", np.zeros(universe.cell_count, dtype=np.float32)
)
system = System(fluid, universe, dt=0.01)

for _ in range(10):
    system.update()

print(system.fluid.positions())
```

Run the maintained DSMC case with:

```bash
python examples/python/main.py cylinder 40
```

Engine selection, wheel construction, NumPy ownership, and binding layout are
documented in the [Python guide](docs/frontends/python.md).

## C++

Build and run the native cylinder example with the TBB backend:

```bash
cmake --preset tbb-gcc-release -DATLAS_EXAMPLES=ON
cmake --build build/tbb-gcc-release --target atlas_example
./build/tbb-gcc-release/examples/cpp/atlas_example cylinder 40
```

Use the `cuda-release` preset for the CUDA backend. Atlas public C++ types are
in the `atlas::` namespace and can be included through `<atlas/atlas.h>`. The
[examples guide](examples/README.md) explains the entry points, templates, and
case layout.

## Interactive

The headless `atlas-interactive` executable accepts newline-delimited JSON
commands and can initialize a simulation from a JSON case:

```bash
cmake -S . -B build/interactive-headless-tbb -G Ninja \
    -DATLAS_DEVICE_SYSTEM=TBB \
    -DATLAS_INTERACTIVE=ON \
    -DATLAS_INTERACTIVE_RENDERING=OFF \
    -DATLAS_PYTHON=OFF \
    -DATLAS_EXAMPLES=OFF \
    -DATLAS_BENCHMARKS=OFF
cmake --build build/interactive-headless-tbb --target atlas-interactive-app
./build/interactive-headless-tbb/src/interactive/atlas-interactive \
    --config examples/interactive/cases/cylinder.json
```

For the native OpenGL window:

```bash
sudo apt-get install -y libgl1-mesa-dev libglew-dev libglfw3-dev libglm-dev
cmake --preset tbb-application-release
cmake --build build/tbb-application-release --target atlas-interactive-example
./build/tbb-application-release/examples/interactive/atlas-interactive-example cylinder 40
```

With rendering enabled, `atlas-interactive` accepts `render_open` and
`render_close` JSONL commands for an existing session. Closing the window leaves
that session alive. The [Interactive guide](docs/frontends/interactive.md)
documents configuration, commands, rendering, and backend state transfer.

## Documentation

- [Examples and adding a case](examples/README.md)
- [Python binding](docs/frontends/python.md)
- [Interactive execution and rendering](docs/frontends/interactive.md)
- [Docker images](docs/operations/docker.md)
- [Core architecture](docs/architecture/overview.md)
- [Core modules](docs/atlas/)
- [Build and contributor documentation](docs/README.md)
- [CI behavior](docs/operations/ci.md)
- [Release operations](docs/operations/releases.md)

## Citation and license

If you use Atlas in research, cite the version used for your results.
[CITATION.cff](CITATION.cff) contains the citation metadata for version 0.1.0:
[doi:10.5281/zenodo.22752179](https://doi.org/10.5281/zenodo.22752179).
The [concept DOI](https://doi.org/10.5281/zenodo.22752178) covers all releases.

Atlas is licensed under [GPL-3.0-or-later](LICENSE).
