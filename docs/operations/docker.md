# Docker guide

Use a runtime image to run simulations, or a development image to edit and
compile Atlas. Both CPU and GPU workflows use the same checkout.

| Goal | Image target | Start here |
|---|---|---|
| Run Python simulations or native applications on a CPU | `tbb` | [Runtime images](#runtime-images) |
| Run simulations on an NVIDIA GPU | `cuda` | [Runtime images](#runtime-images) |
| Develop and compile with a native C++ compiler | `tbb-dev` | [Development setup](#standard-development-environment) |
| Develop and compile with nvcc | `cuda-dev` | [GPU development](#gpu-development) |

## Before you start

Install Docker and clone the repository. The development launcher additionally
needs Python 3 on the host. Run host commands below from the repository root.
For the complete clone/build/run sequence, see the [Quickstart](../../README.md#quickstart).

The container provides compilers and dependencies. Your host Ubuntu version
does not need to match its base image, and you do not need to choose an Ubuntu
version for everyday use. The default base is pinned for reproducible builds;
[build options](#build-options) let you select another supported base.
The development launcher uses Linux UID/GID and filesystem conventions; display
instructions below assume a local Linux desktop.

GPU execution additionally needs a compatible host NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker. CUDA compilation uses nvcc inside the development image;
the host does not need the CUDA toolkit. CPU builds do not require CUDA or a GPU.

## Runtime images

| Target | Base | Installed Atlas engine | Host GPU requirement |
|---|---|---|---|
| `tbb` | Ubuntu | TBB Python runtime plus native executables | OpenGL-capable display only when opening a window |
| `cuda` | NVIDIA CUDA runtime on Ubuntu | CUDA Python runtime plus native executables | NVIDIA GPU/driver and Container Toolkit; display access when opening a window |

The default final target is `tbb`. Its build and runtime do not require the
CUDA toolkit. The CUDA image uses a matching CUDA development image to build
the package; its final image contains runtime libraries instead of the compiler.
Neither image needs a GPU during its build.

Build the desired image from a checkout:

```bash
docker build --target tbb -t atlas:tbb .
docker build --target cuda -t atlas:cuda .
```

The dependency stage downloads Ubuntu packages, Python build tools, and
hash-verified release archives. It installs the C++ dependencies before Atlas
sources enter the build context of the application stages. These commands build
the package; they do not run the Atlas test suites or simulation examples.

Run the included DSMC example:

```bash
docker run --rm atlas:tbb python /opt/atlas/examples/python/main.py cylinder 40
docker run --rm --gpus all atlas:cuda python /opt/atlas/examples/python/main.py cylinder 40
```

The first build downloads and compiles the dependencies and Atlas. Later runs
reuse the image. Rebuild a runtime image when you want it to include source
changes; mounting a checkout does not replace its installed Atlas package.

## Published images

After a successful publication, the `linux/amd64` images can be used without
a source checkout or local compilation:

```bash
docker pull ghcr.io/wontae-lee/atlas-engine-dev:tbb-ubuntu22.04
docker run --rm ghcr.io/wontae-lee/atlas-engine-dev:tbb-ubuntu22.04 python /opt/atlas/examples/python/main.py cylinder 40
docker run --rm --gpus all ghcr.io/wontae-lee/atlas-engine-dev:cuda-ubuntu24.04 python /opt/atlas/examples/python/main.py cylinder 40
```

The engine tags are `tbb-ubuntu22.04`, `tbb-ubuntu24.04`, `cuda-ubuntu22.04`, and
`cuda-ubuntu24.04`. Tags prefixed with the package version, such as
`<version>-tbb-ubuntu22.04`, are also produced. The short `tbb` and `cuda` aliases
use Ubuntu 22.04. A tag is available only after it has been published, and
anonymous pulls require a public package.

For workflow controls, permissions, and tag updates, see
[releases.md](releases.md#docker-publication).

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
to switch backends. See [Python frontend](../frontends/python.md#default-engine) for the selection
rules within a Python process.

## Native executables

Both runtime images provide `/opt/atlas/bin/atlas-interactive` and
`atlas-interactive-example`. Their default command remains `python`; run a
native executable explicitly when needed. `atlas-interactive` accepts JSONL
without opening a display, although the runtime images also include its
rendering support. A display is needed only for `render_open` or the example:

```bash
docker run --rm -i atlas:tbb atlas-interactive \
    --config /opt/atlas/examples/interactive/cases/cylinder.json
```

The `atlas-interactive-example` executable opens the native window. It needs the
host display socket and display environment. For a local X11 session, a typical
TBB invocation is:

```bash
docker run --rm -it \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    atlas:tbb atlas-interactive-example
```

Host display authorization policies still apply. The CUDA server needs
`--gpus all`; the CUDA window additionally needs
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display`. These images do
not yet include a headless OpenGL context or network frame transport. See
[Interactive frontend](../frontends/interactive.md) for the JSON control and rendering APIs.

## Standard development environment

Development uses the `tbb-dev` and `cuda-dev` targets; Core and Python CI use
`tbb-dev`. Ubuntu versions identify the container base, not a requirement on the
host Ubuntu version. The launcher selects its default image without a version
argument; `UBUNTU_VERSION` is an optional override for testing another base.
Both contain CMake, Ninja, GCC/G++, GDB, ccache, Python development headers,
packaging tools, and the graphics development libraries. The host-dependencies
stage installs pinned C++ packages under `/opt/atlas-deps`, then both development
targets reuse that installation. It is built with the same Ubuntu host compiler
and C++20 settings used by both targets. `tbb-dev` does not contain CUDA.

| Target | Purpose |
|---|---|
| `tbb-dev` | Native CPU development, tests, and Python packaging |
| `cuda-dev` | The same dependencies plus nvcc for CUDA and TBB-through-nvcc builds |
| `dev` | Compatibility alias for `cuda-dev` |
| `wheel` | Compatibility packaging alias for `cuda-dev` |
| `tbb`, `cuda` | Separate installed runtimes built from the corresponding development environment |

### Build once, then open a shell

On the host, build the CPU development image:

```bash
python3 scripts/dev.py build tbb
```

Open a development shell whenever you work on Atlas:

```bash
python3 scripts/dev.py tbb
```

Inside that shell, configure and build a native example:

```bash
cmake --preset tbb-gcc-release -DATLAS_EXAMPLES=ON
cmake --build build/tbb-gcc-release --target atlas_example
./build/tbb-gcc-release/examples/cpp/atlas_example cylinder 40
```

Edit files with your host editor. Code remains in the host checkout, mounted at
`/workspace`. Each invocation creates a temporary container as the host UID/GID,
so files it creates belong to you. Use `exit` to return to the host.

### Use Python in the development shell

Development images contain Python build tools; install the current checkout
before using Atlas. Run these commands inside the same development shell:

```bash
python -m pip install --no-build-isolation . -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB
python examples/python/main.py cylinder 40
```

For a CUDA development shell, replace `TBB` with `CUDA` in the install command.
The writable Python environment is `/opt/venv`. Package installations disappear
when the temporary container exits; source files and mounted build results
persist. Reinstall after Python or native binding changes to use the updated
package. To run a preinstalled package, use a [runtime image](#runtime-images).

### Run validation when needed

The launcher also accepts individual commands from the host. For native CPU
validation:

```bash
python3 scripts/dev.py tbb cmake --preset tbb-test
python3 scripts/dev.py tbb cmake --build --preset tbb-test
python3 scripts/dev.py tbb ctest --preset tbb-test
```

For the TBB Python packaging check, installation, and tests in one session:

```bash
python3 scripts/dev.py tbb python scripts/check_python_package.py
```

See the [testing guide](../contributing/testing.md) for suite coverage.

### Build files, caches, and image updates

The launcher preserves these host directories:

| Host path | Container path | Contents |
|---|---|---|
| `build/docker-<backend>-ubuntu<version>/build` | `/workspace/build` | CMake/Ninja build trees |
| `build/docker-<backend>-ubuntu<version>/ccache` | `/tmp/atlas-ccache` | Compiled-object cache |
| `build/docker-<backend>-ubuntu<version>/pip` | `/tmp/atlas-pip-cache` | Python download cache |

The nested build mount keeps container paths stable while separating host,
TBB, CUDA, and Ubuntu build trees. Keep different presets for different compiler
configurations. After changing compilers or the dependency image, start a fresh
CMake build tree. Source changes require only another `cmake --build`; rebuild
the development image when its Dockerfile, package lists, or dependency pins
change. Opening a shell or running a command never rebuilds the image automatically.
To refresh it after dependency changes, run `python3 scripts/dev.py build tbb`
(or `build cuda`) again, then open a new container. Add `--pull` to that build
command when you also want Docker to check for an updated base image.

### Launcher settings

Useful launcher settings:

| Environment variable | Default | Purpose |
|---|---|---|
| `UBUNTU_VERSION` | `22.04` | Image tag and cache-directory variant |
| `ATLAS_DEV_IMAGE` | `atlas:<backend>-dev-ubuntu<version>` | Use another local or published development image |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `2` | Image dependency build and native build parallelism |
| `ATLAS_DOCKER_GPU` | `1` | Set to `0` to compile in the CUDA image without exposing a GPU |
| `ATLAS_DOCKER_DISPLAY` | unset | Set to `1` to forward the host X11 display and available Xauthority file |

### GPU development

Build the CUDA development image on the host, then open its shell:

```bash
python3 scripts/dev.py build cuda
python3 scripts/dev.py cuda
```

Inside the shell, use the `cuda-release` preset for normal native builds or
`cuda-test` for validation. To run validation directly from the host:

```bash
python3 scripts/dev.py cuda cmake --preset cuda-test
python3 scripts/dev.py cuda cmake --build --preset cuda-test
python3 scripts/dev.py cuda ctest --preset cuda-test
```

To compile without exposing a GPU, open the shell with
`ATLAS_DOCKER_GPU=0 python3 scripts/dev.py cuda`. Running CUDA simulations or
tests still requires GPU access.

### Display and IDE setup

For a development shell with access to the host X11 display:

```bash
ATLAS_DOCKER_DISPLAY=1 python3 scripts/dev.py cuda
```

CUDA execution requires the host NVIDIA driver and Container Toolkit. The
launcher supplies `--gpus all`; the image supplies nvcc. Display forwarding also
requests NVIDIA graphics/display capabilities. X11 authorization must permit
access as the host user. On a Wayland desktop this requires XWayland; the launcher
does not provide a display server.

CLion can use these same development images as Docker toolchains. Select the
image, mount the checkout, and use a matching TBB/CUDA preset. Ensure the IDE
retains its build directory across container sessions; its mount layout may
differ from the launcher. Configure GPU access for CUDA test execution.

## Build options

| Build argument | Default | Purpose |
|---|---|---|
| `UBUNTU_VERSION` | `22.04` | Ubuntu base version; `24.04` is also configured |
| `CUDA_VERSION` | `12.9.2` | CUDA version for the CUDA development and runtime bases |
| `CMAKE_CUDA_ARCHITECTURES` | `75-real;80-real;86-real;89-real;90` | CUDA machine-code targets and compute 90 PTX |
| `BUILD_JOBS` | `2` | Parallel compile jobs during package and runtime builds |

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

## Build caches and publication

The Dockerfile installs dependencies before copying Atlas sources. Source edits
therefore reuse that dependency layer. Runtime build stages additionally use
BuildKit cache mounts for pip, ccache, and their backend-specific CMake build
trees. Caches can be recreated from the declared build inputs; removing them can
make subsequent builds slower.

GitHub Actions imports and exports image layers through the GitHub Actions
cache. Core/Python CI also explicitly preserves its mounted ccache directory.
BuildKit cache-mount contents are local to the builder and are not exported by
the image-layer `type=gha` cache; a fresh hosted publication runner may need to
compile Atlas again after its source layer changes. The prebuilt dependency
layer can still be reused. No compile-time improvement is claimed without a
measurement on matching workloads and cache states.

The public PyPI wheels use a manylinux build container with the same dependency
installer and pins. That environment additionally builds oneTBB from its pinned
release; Ubuntu development images alone do not establish manylinux compatibility.
See [Python packaging](../frontends/python.md#packaging-a-wheel-and-installing-it-later).

## Troubleshooting

| Symptom | What to check |
|---|---|
| Development image is missing | Run `python3 scripts/dev.py build tbb` or `build cuda` with the same launcher settings used to enter it. |
| Python cannot import Atlas in a development shell | Install the checkout inside that session; a previous temporary container's installation is gone. |
| Source edits are absent from a runtime image | Rebuild the runtime image. For ongoing development, use the mounted development environment. |
| CMake refers to an old compiler or dependency path | Use a fresh CMake build tree after changing the compiler or dependency image. |
| A viewer cannot open a display | Check `DISPLAY`, the X11 socket, and Xauthority access. Wayland sessions need XWayland. |
| CUDA cannot see a GPU | Check the host NVIDIA driver and Container Toolkit configuration; use `--gpus all` for runtime images. |

If Docker reports `failed to discover GPU vendor from CDI`, check that
`nvidia-ctk` is installed on the host. Follow NVIDIA's installation guide, then
configure Docker with `sudo nvidia-ctk runtime configure --runtime=docker` and
restart its daemon with `sudo systemctl restart docker`. Restarting Docker may
interrupt other containers. Rebuilding Atlas does not repair a missing host GPU
runtime.
