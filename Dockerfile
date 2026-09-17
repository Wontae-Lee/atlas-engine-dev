# ============================================================
# Atlas Engine — Docker images
#
# Atlas selects its execution backend at configure time:
#
#   -DATLAS_DEVICE_SYSTEM=TBB   → CPU build (std::vector + TBB)
#   -DATLAS_DEVICE_SYSTEM=CUDA  → GPU build (Thrust + CUDA)
#
# TBB uses native C/C++ compilers by default; tbb-gcc-* presets
# select gcc/g++ explicitly, without requiring the CUDA toolkit.
# ATLAS_HOST_COMPILER=nvcc also supports TBB builds through nvcc.
# CUDA requires nvcc; host-only code keeps its C/C++ compilers.
#
# A GPU is NOT required to build either variant; it is required
# only to run the CUDA variant.
#
# Stages:
#
#   dev      Development environment. Full toolchain, no sources.
#            This is the image contributors work inside:
#
#              docker build --target dev -t atlas-dev .
#              docker run --rm -it -v "$PWD":/workspace atlas-dev
#              # inside the container:
#              cmake --preset tbb-debug
#              cmake --build build/tbb-debug -j$(nproc)
#
#   builder  Compiles the repository with a selected preset.
#
#              docker build --target builder \
#                  --build-arg ATLAS_PRESET=cuda-release .
#
#   runtime  Thin image holding only built artifacts and
#            runtime libraries. Default final stage.
# ============================================================

# ============================================================
# dev — development environment
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev
LABEL authors="Wontae Lee"

ENV DEBIAN_FRONTEND=noninteractive

# Core toolchain and library dependencies:
#   - build-essential / cmake / ninja-build : host compiler + build system
#     (nvcc itself ships with the CUDA base image)
#   - libtbb-dev  : native TBB backend and Thrust host system for CUDA
#   - python3-dev : required by the optional nanobind Python bindings
#   - git / gdb / vim / pkg-config : everyday development utilities
RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        gdb \
        vim \
        pkg-config \
        libtbb-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENTRYPOINT ["bash"]

# ============================================================
# wheel — dev environment plus Python packaging tools, for
#         building self-contained (auditwheel-repaired) wheels:
#
#   docker build --target wheel -t atlas-wheel .
#   docker run --rm -v "$PWD":/workspace -w /workspace atlas-wheel \
#       -c 'bash scripts/build_wheels.sh'
#
# Produces dist/tbb/*.whl and dist/cuda/*.whl, each bundling its
# runtime libraries (libtbb, libcudart, libstdc++, ...).
# ============================================================
FROM dev AS wheel

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Bake the packaging toolchain into the image so build_wheels.sh needs no
# network at run time (its pip install then becomes a no-op).
RUN python3 -m pip install --no-cache-dir \
        "scikit-build-core>=0.10" "nanobind>=2.0" build auditwheel patchelf

WORKDIR /workspace

ENTRYPOINT ["bash"]

# ============================================================
# builder — compile the repository inside the dev environment
# ============================================================
FROM dev AS builder

# Configure preset to build. One of:
#   tbb-release / tbb-debug / tbb-gcc-release / tbb-gcc-debug
#   tbb-nvcc-debug / cuda-release / cuda-debug
ARG ATLAS_PRESET=tbb-release

WORKDIR /app
ADD . /app

RUN cmake --preset ${ATLAS_PRESET} && \
    cmake --build build/${ATLAS_PRESET} -j"$(nproc)"

# ============================================================
# runtime — thin image with runtime dependencies only
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
LABEL authors="Wontae Lee"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
        libtbb2 \
    && rm -rf /var/lib/apt/lists/*

# Build artifacts stay under /app/build in the builder stage;
# copy them over for direct execution or packaging.
COPY --from=builder /app/build /opt/atlas/build

WORKDIR /work

ENTRYPOINT ["bash"]
