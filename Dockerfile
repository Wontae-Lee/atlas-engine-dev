ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.9.2

FROM ubuntu:${UBUNTU_VERSION} AS tbb-dev

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake ninja-build git gdb vim pkg-config \
        libtbb-dev python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    CC=gcc \
    CXX=g++

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip

WORKDIR /workspace
ENTRYPOINT ["bash"]

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS cuda-dev

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake ninja-build git gdb vim pkg-config \
        libtbb-dev python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    CC=gcc \
    CXX=g++

RUN python3 -m venv "${VIRTUAL_ENV}" && \
    python -m pip install --no-cache-dir --upgrade pip

WORKDIR /workspace
ENTRYPOINT ["bash"]

FROM cuda-dev AS dev

FROM cuda-dev AS wheel

RUN python -m pip install --no-cache-dir \
    "scikit-build-core>=0.10" "nanobind>=2.0" build auditwheel patchelf

FROM tbb-dev AS tbb-builder

ARG BUILD_JOBS=2
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

WORKDIR /src
COPY . .

RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels . \
        -C cmake.define.ATLAS_DEVICE_SYSTEM=TBB \
        -C cmake.define.ATLAS_HOST_COMPILER=native \
        -C cmake.define.BUILD_TESTING=OFF \
        -C cmake.define.BUILD_SHARED_LIBS=OFF && \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels atlas-engine && \
    rm -rf /wheels

FROM cuda-dev AS cuda-builder

ARG BUILD_JOBS=2
ARG CMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}

WORKDIR /src
COPY . .

RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels . \
        -C cmake.define.ATLAS_DEVICE_SYSTEM=CUDA \
        -C cmake.define.ATLAS_HOST_COMPILER=native \
        -C "cmake.define.CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" \
        -C cmake.define.BUILD_TESTING=OFF \
        -C cmake.define.BUILD_SHARED_LIBS=OFF && \
    python -m pip install --no-cache-dir --no-index --find-links=/wheels atlas-engine && \
    rm -rf /wheels

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS cuda

LABEL org.opencontainers.image.title="Atlas Engine (CUDA)" \
    org.opencontainers.image.source="https://github.com/Wontae-Lee/atlas-engine-dev" \
    org.opencontainers.image.licenses="GPL-3.0-or-later"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates python3 libtbb12 libstdc++6 zlib1g && \
    rm -rf /var/lib/apt/lists/*

COPY --from=cuda-builder /opt/venv /opt/venv
COPY examples/python /opt/atlas/examples/python
COPY assets /opt/atlas/assets

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    ATLAS_DEFAULT_ENGINE=cuda \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["python"]

FROM ubuntu:${UBUNTU_VERSION} AS tbb

LABEL org.opencontainers.image.title="Atlas Engine (TBB)" \
    org.opencontainers.image.source="https://github.com/Wontae-Lee/atlas-engine-dev" \
    org.opencontainers.image.licenses="GPL-3.0-or-later"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates python3 libtbb12 libstdc++6 zlib1g && \
    rm -rf /var/lib/apt/lists/*

COPY --from=tbb-builder /opt/venv /opt/venv
COPY examples/python /opt/atlas/examples/python
COPY assets /opt/atlas/assets

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    ATLAS_DEFAULT_ENGINE=tbb \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
CMD ["python"]
