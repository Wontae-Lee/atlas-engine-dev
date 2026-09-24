<p align="center">
  <img src="docs/atlas-engine-logo.png" alt="Atlas Engine logo" width="1004">
</p>

# Atlas Engine

[![DOI](https://zenodo.org/badge/1021472310.svg)](https://doi.org/10.5281/zenodo.22752178)
[![Core CI](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-core.yml/badge.svg?branch=main)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-core.yml)
[![Python CI](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/Wontae-Lee/atlas-engine-dev/actions/workflows/ci-python.yml)
[![PyPI](https://img.shields.io/pypi/v/atlas-engine.svg)](https://pypi.org/project/atlas-engine/)

Atlas is a C++20 engine for simulating rarefied-gas flows with direct simulation
Monte Carlo (DSMC). Run simulations on a CPU with TBB or an NVIDIA GPU with CUDA,
write them in Python or C++, and inspect them with the native Interactive viewer.

It includes collision models, analytic and OBJ mesh geometry, particle sources
and boundaries, saved simulation snapshots, and NumPy access to particle data.

## Quickstart

Start with the CPU example. You need Git and a working Docker installation;
no host C++ compiler, Python environment, or CUDA toolkit is required.

```bash
git clone https://github.com/Wontae-Lee/atlas-engine-dev.git
cd atlas-engine-dev
docker build --target tbb -t atlas:tbb .
docker run --rm atlas:tbb python /opt/atlas/examples/python/main.py cylinder 40
```

The first build downloads dependencies and compiles Atlas. The example runs
40 simulation steps and prints particle counts followed by a `completed:` line.
Run the last command again to reuse the image.

Your host Ubuntu version does not need to match the container's version. The
image supplies the environment needed to run Atlas.

### Run on an NVIDIA GPU

Prepare the host NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then build and run the CUDA image:

```bash
docker build --target cuda -t atlas:cuda .
docker run --rm --gpus all atlas:cuda \
    python /opt/atlas/examples/python/main.py cylinder 40
```

The GPU is needed when running the simulation. CUDA compilation takes place
inside Docker. See the [Docker guide](docs/operations/docker.md) for GPU setup,
published images, and build options.

## Write a Python simulation

Save this as `simulation.py` in your current directory:

```python
import numpy as np
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

Run it using the CPU image built above:

```bash
docker run --rm -v "$PWD":/workspace atlas:tbb python simulation.py
```

To use the GPU, change the image to `atlas:cuda` and add `--gpus all`. Each image
selects its installed engine automatically. Files written under `/workspace`
are saved in your mounted host directory.

For an interactive Python prompt:

```bash
docker run --rm -it atlas:tbb
```

The package is named `atlas-engine` and imported as `atlas`.
The [Python guide](docs/frontends/python.md) covers wheel installation, engine
selection, and the API. The [examples](examples/README.md) show a complete DSMC
flow with particle emission and collisions.

## Control a simulation or open the viewer

Start the native JSONL server with the bundled cylinder case:

```bash
docker run --rm -i atlas:tbb atlas-interactive \
    --config /opt/atlas/examples/interactive/cases/cylinder.json
```

After the startup response creates session `1`, send these lines to advance it
and shut down:

```json
{"command":"step","session_id":1,"payload":{"step_count":3}}
{"command":"shutdown"}
```

The native viewer displays particles and scene geometry in an OpenGL window.
Follow the [display setup](docs/operations/docker.md#native-executables) to run it.
See the [Interactive guide](docs/frontends/interactive.md) for commands and JSON
configuration, or start from an [example case](examples/README.md).

## Develop Atlas

From your checkout, use Python 3 on the host to build and enter the standard
CPU development environment:

```bash
python3 scripts/dev.py build tbb
python3 scripts/dev.py tbb
```

Inside that shell, build the native example:

```bash
cmake --preset tbb-gcc-release -DATLAS_EXAMPLES=ON
cmake --build build/tbb-gcc-release --target atlas_example
./build/tbb-gcc-release/examples/cpp/atlas_example cylinder 40
```

Source edits and build results persist on the host. For GPU development, use
`cuda` with the launcher and the `cuda-release` CMake preset.
The [development guide](docs/operations/docker.md#standard-development-environment)
covers Python installation, reusable build caches, and IDE setup.

## Learn more

| Goal | Guide |
|---|---|
| Run Docker images, mount files, or use a GPU | [Docker](docs/operations/docker.md) |
| Build a simulation in Python or C++ | [Examples and templates](examples/README.md) |
| Use Python objects and NumPy arrays | [Python API](docs/frontends/python.md) |
| Control sessions or render a simulation | [Interactive](docs/frontends/interactive.md) |
| Build, test, or contribute to Atlas | [Developer documentation](docs/README.md) |
| Understand the engine | [Architecture](docs/architecture/overview.md) and [Core modules](docs/atlas/) |

## Citation and license

If you use Atlas in research, cite the version used for your results.
[CITATION.cff](CITATION.cff) contains the citation metadata for version 0.1.0:
[doi:10.5281/zenodo.22752179](https://doi.org/10.5281/zenodo.22752179).
The [concept DOI](https://doi.org/10.5281/zenodo.22752178) covers all releases.

Atlas is licensed under [GPL-3.0-or-later](LICENSE).
