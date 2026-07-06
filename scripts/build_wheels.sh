#!/usr/bin/env bash
# Build self-contained (auditwheel-repaired) Atlas wheels for both backends.
#
# Runs entirely inside the atlas-dev / atlas-wheel Docker image:
#
#   docker build --target wheel -t atlas-wheel .
#   docker run --rm -v "$PWD":/workspace -w /workspace atlas-wheel \
#       -c 'bash scripts/build_wheels.sh'
#
# Output: dist/tbb/*.whl  and  dist/cuda/*.whl — each bundles its runtime
# libraries (libtbb, libcudart, libstdc++, ...) via auditwheel. The CUDA
# driver (libcuda.so.1) is deliberately NOT bundled; it must come from the
# host's NVIDIA driver at run time.
set -euo pipefail

# Packaging tools. No-ops when the image already has them (the `wheel` stage).
python3 -m pip install --quiet --no-input \
    "scikit-build-core>=0.10" "nanobind>=2.0" build auditwheel patchelf

BACKENDS="${ATLAS_WHEEL_BACKENDS:-TBB CUDA}"

for backend in ${BACKENDS}; do
    low="$(echo "${backend}" | tr '[:upper:]' '[:lower:]')"
    echo "==================== building ${backend} wheel ===================="

    rm -rf dist/_raw "build/wheel-${low}"
    python3 -m build --wheel --no-isolation -o dist/_raw \
        -C "cmake.define.ATLAS_DEVICE_SYSTEM=${backend}" \
        -C "build-dir=build/wheel-${low}"

    echo "-------------------- auditwheel repair (${backend}) --------------------"
    # --exclude libcuda.so.1 : the CUDA driver is host-provided, never bundled.
    auditwheel repair dist/_raw/*.whl -w "dist/${low}" \
        --exclude libcuda.so.1
    rm -rf dist/_raw
done

echo "==================== done ===================="
find dist -name '*.whl' -printf '%s\t%p\n' 2>/dev/null | \
    awk -F'\t' '{printf "%6.1f MB  %s\n", $1/1048576, $2}'
