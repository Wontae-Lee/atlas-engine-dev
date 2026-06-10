#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
splishsplash_bin="${repo_root}/benchmarks/splishsplash/bin/SPHSimulator"

scene_file="cylinder.smoke.json"
stop_at="0.005"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            scene_file="cylinder.json"
            stop_at="0.05"
            shift
            ;;
        --smoke)
            scene_file="cylinder.smoke.json"
            stop_at="0.005"
            shift
            ;;
        *)
            echo "Usage: $0 [--smoke|--full]" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${splishsplash_bin}" ]]; then
    echo "SPlisHSPlasH executable not found: ${splishsplash_bin}" >&2
    echo "Build it with: cmake --build --preset build-tbb-debug --target atlas_benchmark_splishsplash_external" >&2
    exit 1
fi

cd "${script_dir}"
mkdir -p output

exec "${splishsplash_bin}" \
    "${script_dir}/${scene_file}" \
    --no-gui \
    --no-initial-pause \
    --no-cache \
    --stopAt "${stop_at}" \
    --output-dir "${script_dir}/output/${scene_file%.json}"
