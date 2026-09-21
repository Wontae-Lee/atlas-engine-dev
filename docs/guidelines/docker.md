# Docker Images

The root [`Dockerfile`](../../Dockerfile) provides separate CPU and GPU images
with Python, NumPy, and the Atlas package installed. Images are built from the
current checkout. Local build commands use `atlas:*` tags; the publication
workflow uploads separate engine/Ubuntu tags to GHCR.

## Published images

After a successful publication, the `linux/amd64` images can be used without
a source checkout or local compilation:

```bash
docker pull ghcr.io/wontae-lee/atlas-engine-dev:tbb-ubuntu22.04
docker run --rm ghcr.io/wontae-lee/atlas-engine-dev:tbb-ubuntu22.04 python /opt/atlas/examples/python/cylinder.py
docker run --rm --gpus all ghcr.io/wontae-lee/atlas-engine-dev:cuda-ubuntu24.04 python /opt/atlas/examples/python/cylinder.py
```

The engine tags are `tbb-ubuntu22.04`, `tbb-ubuntu24.04`, `cuda-ubuntu22.04`, and
`cuda-ubuntu24.04`. Tags prefixed with the package version, such as
`0.1.0-tbb-ubuntu22.04`, are also produced. The short `tbb` and `cuda` aliases
use Ubuntu 22.04. A tag is available only after it has been published, and
anonymous pulls require a public package.

The same mounts and Python commands described below work with these image
names. For workflow controls, permissions, and tag updates, see
[releases.md](releases.md#build-or-publish-docker-images).

## Runtime images

| Target | Base | Installed Atlas engine | Host GPU requirement |
|---|---|---|---|
| `tbb` | Ubuntu | `tbb`, selected by default | None |
| `cuda` | NVIDIA CUDA runtime on Ubuntu | `cuda`, selected by default | Compatible NVIDIA GPU and driver, plus NVIDIA Container Toolkit |

The default final target is `tbb`. Its build and runtime do not require the
CUDA toolkit. The CUDA image uses a matching CUDA development image to build
the package; its final image contains runtime libraries instead of the compiler.
Neither image needs a GPU during its build.

Build the desired image from a checkout with its submodules populated:

```bash
git submodule update --init --recursive
docker build --target tbb -t atlas:tbb .
docker build --target cuda -t atlas:cuda .
```

Builds download Ubuntu packages, Python build dependencies, and CMake
dependencies such as Abseil. These commands build the package; they do not run
the Atlas test suites or simulation examples.

Run the included DSMC example:

```bash
docker run --rm atlas:tbb python /opt/atlas/examples/python/cylinder.py
docker run --rm --gpus all atlas:cuda python /opt/atlas/examples/python/cylinder.py
```

CUDA execution requires a compatible NVIDIA driver on the host and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker. The host does not need a separate CUDA toolkit.

## Python and files

Python and the package live in `/opt/venv`, which is already on `PATH`.
`python` starts that interpreter, and `pip` installs into that environment.
The default command is `python`, so an interactive session needs only:

```bash
docker run --rm -it atlas:tbb
docker run --rm -it --gpus all atlas:cuda
```

The working directory is `/workspace`. Mount your script and data directory
there, then pass a normal Python command:

```bash
docker run --rm -v "$PWD":/workspace atlas:tbb python simulation.py
docker run --rm --gpus all -v "$PWD":/workspace atlas:cuda python simulation.py
```

Outputs written under `/workspace` persist in that mounted host directory.
Bundled Python examples are under `/opt/atlas/examples/python`, and the
repository's mesh assets are under `/opt/atlas/assets`. A shell is available
with `docker run --rm -it atlas:tbb bash`.

Each image sets `ATLAS_DEFAULT_ENGINE` to its installed engine. Changing that
environment variable cannot add the other engine: use the corresponding image
to switch backends. See [python.md](python.md#default-engine) for the selection
rules within a Python process.

## Build options

| Build argument | Default | Purpose |
|---|---|---|
| `UBUNTU_VERSION` | `22.04` | Ubuntu base version; `24.04` is also configured |
| `CUDA_VERSION` | `12.9.2` | CUDA version for the CUDA development and runtime bases |
| `CMAKE_CUDA_ARCHITECTURES` | `75-real;80-real;86-real;89-real;90` | CUDA machine-code targets and compute 90 PTX |
| `BUILD_JOBS` | `2` | Parallel compile jobs during wheel builds |

Ubuntu 24.04 images use the same targets:

```bash
docker build --target tbb --build-arg UBUNTU_VERSION=24.04 -t atlas:tbb-ubuntu24.04 .
docker build --target cuda --build-arg UBUNTU_VERSION=24.04 -t atlas:cuda-ubuntu24.04 .
```

When overriding `CUDA_VERSION`, choose an NVIDIA tag available for the selected
Ubuntu version in both `devel` and `runtime` variants. The default architecture
list makes GPU targets explicit even when building without a GPU. To build for
a specific target, override it with CMake's architecture syntax:

```bash
docker build --target cuda \
    --build-arg CMAKE_CUDA_ARCHITECTURES=89 \
    --build-arg BUILD_JOBS=4 \
    -t atlas:cuda-sm89 .
```

Select architectures supported by the chosen CUDA toolkit and the GPUs that
will run the image. The architecture list affects build time and image size.

## Development and packaging images

| Target | Contents |
|---|---|
| `tbb-dev` | GCC/G++, TBB development libraries, CMake, Ninja, Python development headers, and a Python virtual environment |
| `cuda-dev` | The same host development tools plus the CUDA toolkit and nvcc |
| `dev` | Compatibility alias for `cuda-dev` |
| `wheel` | CUDA development tools plus the wheel packaging/repair tools used by `scripts/build_wheels.sh` |

These targets provide toolchains and accept mounted source code; they do not
have Atlas preinstalled. Their entry point is Bash:

```bash
docker build --target tbb-dev -t atlas:tbb-dev .
docker run --rm -it -v "$PWD":/workspace atlas:tbb-dev
docker build --target cuda-dev -t atlas:cuda-dev .
docker run --rm -it --gpus all -v "$PWD":/workspace atlas:cuda-dev
```

Add GPU access only when running CUDA code. For CMake commands, see
[build-and-test.md](build-and-test.md); for producing redistributable wheels,
see [python.md](python.md#packaging-a-wheel-and-installing-it-later).
