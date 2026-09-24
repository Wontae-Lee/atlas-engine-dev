# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.9.2

FROM ubuntu:${UBUNTU_VERSION} AS host-dependencies
ARG UBUNTU_VERSION
COPY docker/packages.txt /tmp/atlas-packages.txt
RUN for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=3 update && \
           DEBIAN_FRONTEND=noninteractive xargs apt-get -o Acquire::Retries=3 install -y --no-install-recommends < /tmp/atlas-packages.txt; then \
            break; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    CC=gcc CXX=g++ \
    CMAKE_PREFIX_PATH=/opt/atlas-deps \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/tmp/atlas-ccache
COPY docker/requirements-build.txt /opt/atlas/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install -r /opt/atlas/requirements-build.txt && \
    chmod -R a+w "${VIRTUAL_ENV}"
COPY cmake/dependencies /tmp/atlas-deps/cmake/dependencies
COPY scripts/install_dependencies.py /tmp/atlas-deps/scripts/install_dependencies.py
ARG BUILD_JOBS=2
RUN --mount=type=cache,target=/tmp/atlas-dependency-downloads \
    --mount=type=cache,target=/tmp/atlas-ccache,sharing=locked \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    python /tmp/atlas-deps/scripts/install_dependencies.py \
        -DATLAS_DEPS_DOWNLOAD_DIR=/tmp/atlas-dependency-downloads && \
    rm -rf /tmp/atlas-deps
COPY licenses /opt/atlas/licenses

FROM host-dependencies AS tbb-dev
WORKDIR /workspace
ENTRYPOINT ["bash"]

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS cuda-dev
ARG UBUNTU_VERSION
COPY docker/packages.txt /tmp/atlas-packages.txt
RUN for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=3 update && \
           DEBIAN_FRONTEND=noninteractive xargs apt-get -o Acquire::Retries=3 install -y --no-install-recommends < /tmp/atlas-packages.txt; then \
            break; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*
COPY --from=host-dependencies /opt/atlas-deps /opt/atlas-deps
COPY --from=host-dependencies /opt/venv /opt/venv
COPY --from=host-dependencies /opt/atlas /opt/atlas
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    CC=gcc CXX=g++ \
    CMAKE_PREFIX_PATH=/opt/atlas-deps \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/tmp/atlas-ccache
WORKDIR /workspace
ENTRYPOINT ["bash"]

FROM cuda-dev AS dev
FROM cuda-dev AS wheel

FROM tbb-dev AS tbb-builder
ARG UBUNTU_VERSION
ARG BUILD_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/tmp/atlas-ccache,sharing=locked \
    --mount=type=cache,id=atlas-tbb-wheel-${UBUNTU_VERSION},target=/src/build,sharing=locked \
    python -m pip wheel --no-build-isolation --wheel-dir /wheels . pip==25.1.1 \
        -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB \
        -C cmake.define.ATLAS_HOST_COMPILER=native \
        -C cmake.define.BUILD_TESTING=OFF \
        -C cmake.define.BUILD_SHARED_LIBS=OFF && \
    rm -rf /opt/venv && \
    /usr/bin/python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --no-index --find-links=/wheels pip==25.1.1 atlas-engine && \
    rm -rf /wheels

FROM cuda-dev AS cuda-builder
ARG UBUNTU_VERSION
ARG BUILD_JOBS=2
ARG CMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/tmp/atlas-ccache,sharing=locked \
    --mount=type=cache,id=atlas-cuda-wheel-${UBUNTU_VERSION},target=/src/build,sharing=locked \
    python -m pip wheel --no-build-isolation --wheel-dir /wheels . pip==25.1.1 \
        -C cmake.define.ATLAS_DEVICE_SYSTEM=CUDA \
        -C cmake.define.ATLAS_HOST_COMPILER=native \
        -C "cmake.define.CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" \
        -C cmake.define.BUILD_TESTING=OFF \
        -C cmake.define.BUILD_SHARED_LIBS=OFF && \
    rm -rf /opt/venv && \
    /usr/bin/python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --no-index --find-links=/wheels pip==25.1.1 atlas-engine && \
    rm -rf /wheels

FROM tbb-dev AS tbb-application-builder
ARG UBUNTU_VERSION
ARG BUILD_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/tmp/atlas-ccache,sharing=locked \
    --mount=type=cache,id=atlas-tbb-application-${UBUNTU_VERSION},target=/src/build,sharing=locked \
    cmake --preset tbb-application-release && \
    cmake --build build/tbb-application-release \
        --target atlas-interactive-app atlas-interactive-example && \
    cmake --install build/tbb-application-release \
        --prefix /opt/atlas --component interactive

FROM cuda-dev AS cuda-application-builder
ARG UBUNTU_VERSION
ARG BUILD_JOBS=2
ARG CMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/tmp/atlas-ccache,sharing=locked \
    --mount=type=cache,id=atlas-cuda-application-${UBUNTU_VERSION},target=/src/build,sharing=locked \
    cmake --preset cuda-application-release -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" && \
    cmake --build build/cuda-application-release \
        --target atlas-interactive-app atlas-interactive-example && \
    cmake --install build/cuda-application-release \
        --prefix /opt/atlas --component interactive

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS cuda

LABEL org.opencontainers.image.title="Atlas Engine (CUDA)" \
    org.opencontainers.image.source="https://github.com/Wontae-Lee/atlas-engine-dev" \
    org.opencontainers.image.licenses="GPL-3.0-or-later"

RUN for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=3 update && \
           DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
               ca-certificates python3 libgl1 libopengl0 libglew2.2 libglfw3 libtbb12 libstdc++6 zlib1g; then \
            break; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*

COPY --from=cuda-builder /opt/venv /opt/venv
COPY --from=cuda-application-builder /opt/atlas /opt/atlas
COPY licenses /opt/atlas/licenses
COPY examples/python /opt/atlas/examples/python
COPY assets /opt/atlas/assets

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/atlas/bin:${VIRTUAL_ENV}/bin:${PATH}" \
    ATLAS_DEFAULT_ENGINE=cuda \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

WORKDIR /workspace
CMD ["python"]

FROM ubuntu:${UBUNTU_VERSION} AS tbb

LABEL org.opencontainers.image.title="Atlas Engine (TBB)" \
    org.opencontainers.image.source="https://github.com/Wontae-Lee/atlas-engine-dev" \
    org.opencontainers.image.licenses="GPL-3.0-or-later"

RUN for attempt in 1 2 3; do \
        if apt-get -o Acquire::Retries=3 update && \
           DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
               ca-certificates python3 libgl1 libopengl0 libglew2.2 libglfw3 libtbb12 libstdc++6 zlib1g; then \
            break; \
        fi; \
        if [ "$attempt" -eq 3 ]; then exit 1; fi; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
    done && \
    rm -rf /var/lib/apt/lists/*

COPY --from=tbb-builder /opt/venv /opt/venv
COPY --from=tbb-application-builder /opt/atlas /opt/atlas
COPY licenses /opt/atlas/licenses
COPY examples/python /opt/atlas/examples/python
COPY assets /opt/atlas/assets

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/atlas/bin:${VIRTUAL_ENV}/bin:${PATH}" \
    ATLAS_DEFAULT_ENGINE=tbb \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["python"]
