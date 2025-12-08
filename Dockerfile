# ============================================================
# Builder stage (CUDA + full development toolchain)
#   - Uses official NVIDIA CUDA devel image
#   - Compiles and installs atlas-core into /usr/local
# ============================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder
LABEL authors="Wontae Lee"

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Build-time option:
#   - ATLAS_USE_CUDA=ON  → enable CUDA in CMake
#   - ATLAS_USE_CUDA=OFF → build without CUDA (still using CUDA base image)
ARG ATLAS_USE_CUDA=ON

# ------------------------------------------------------------
# Install build dependencies
#   - build-essential, cmake, ninja-build: core C/C++ toolchain
#   - git, gdb, vim: common developer utilities
#   - pkg-config: library detection
#   - libtbb-dev: TBB development package (headers + libs)
# ------------------------------------------------------------
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
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Copy source tree into /app
# ------------------------------------------------------------
WORKDIR /app
ADD . /app

# ------------------------------------------------------------
# Configure, build, and install using CMake presets
#   - docker-release preset is defined in CMakePresets.json
#   - ATLAS_USE_CUDA is forwarded as a cache variable override
# ------------------------------------------------------------
RUN cmake --preset docker-release -DATLAS_USE_CUDA=${ATLAS_USE_CUDA} && \
    cmake --build --preset docker-release -j"$(nproc)" && \
    cmake --install --preset docker-release

# ============================================================
# Runtime stage (thin image, only what is needed to run)
#   - Uses official NVIDIA CUDA runtime image
#   - Receives installed artifacts from builder stage
# ============================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime
LABEL authors="Wontae Lee"

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Install runtime dependencies only
#   - libtbb2: shared TBB runtime library used by atlas-core
#   - No compiler / cmake / git / etc. in this stage
# ------------------------------------------------------------
RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
        libtbb2 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Copy installed atlas-core (and any executables) from builder
#   - This assumes CMAKE_INSTALL_PREFIX=/usr/local
# ------------------------------------------------------------
COPY --from=builder /usr/local /usr/local

# Optional: working directory for runtime containers
WORKDIR /work

# ------------------------------------------------------------
# Default entrypoint
#   - Drop into an interactive shell by default.
#   - Override with `docker run --entrypoint <binary>` as needed.
# ------------------------------------------------------------
ENTRYPOINT ["bash"]
