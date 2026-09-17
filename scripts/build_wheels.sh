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

want_tbb=false
want_cuda=false
read -r -a requested_backends <<< "${ATLAS_WHEEL_BACKENDS:-TBB CUDA}"
for backend in "${requested_backends[@]}"; do
    case "${backend^^}" in
        TBB) want_tbb=true ;;
        CUDA) want_cuda=true ;;
        *)
            printf 'Unknown wheel backend: %s (expected TBB or CUDA)\n' "${backend}" >&2
            exit 1
            ;;
    esac
done

# Packaging tools. No-ops when the image already has them (the `wheel` stage).
python3 -m pip install --quiet --no-input \
    "scikit-build-core>=0.10" "nanobind>=2.0" build auditwheel patchelf

wheel_stage="$(mktemp -d "${TMPDIR:-/tmp}/atlas-wheels.XXXXXX")"
trap 'rm -rf "${wheel_stage}"' EXIT

build_wheel() {
    local backend="$1"
    shift
    local low="${backend,,}"
    printf 'Building %s wheel\n' "${backend}"

    python3 -m build --wheel --no-isolation -o "${wheel_stage}/${low}" \
        -C "cmake.define.ATLAS_DEVICE_SYSTEM=${backend}" \
        -C "build-dir=build/wheel-${low}" \
        "$@"
}

repair_wheel() {
    local low="$1"
    printf 'Repairing %s wheel\n' "${low}"
    auditwheel repair "${wheel_stage}/${low}"/*.whl -w "dist/${low}" \
        --exclude libcuda.so.1
}

build_wheel TBB -C "cmake.define.ATLAS_PYTHON_TBB_EXTENSION="

if "${want_tbb}"; then
    repair_wheel tbb
fi

if "${want_cuda}"; then
    tbb_extension="$(python3 - "${wheel_stage}/tbb" "${wheel_stage}/native" <<'PY'
import pathlib
import sys
import zipfile

wheel_dir = pathlib.Path(sys.argv[1])
native_dir = pathlib.Path(sys.argv[2])
wheels = list(wheel_dir.glob("*.whl"))
if len(wheels) != 1:
    raise RuntimeError(f"Expected one TBB wheel in {wheel_dir}, found {len(wheels)}")

with zipfile.ZipFile(wheels[0]) as wheel:
    extensions = [
        name for name in wheel.namelist()
        if pathlib.PurePosixPath(name).parent == pathlib.PurePosixPath("atlas")
        and pathlib.PurePosixPath(name).name.startswith("_core_tbb.")
        and name.endswith(".so")
    ]
    if len(extensions) != 1:
        raise RuntimeError(f"Expected one _core_tbb extension, found {extensions}")
    native_dir.mkdir()
    extension = native_dir / pathlib.PurePosixPath(extensions[0]).name
    extension.write_bytes(wheel.read(extensions[0]))
    print(extension.resolve())
PY
)"
    build_wheel CUDA -C "cmake.define.ATLAS_PYTHON_TBB_EXTENSION=${tbb_extension}"
    repair_wheel cuda
fi

echo "==================== done ===================="
find dist -name '*.whl' -printf '%s\t%p\n' 2>/dev/null | \
    awk -F'\t' '{printf "%6.1f MB  %s\n", $1/1048576, $2}'
