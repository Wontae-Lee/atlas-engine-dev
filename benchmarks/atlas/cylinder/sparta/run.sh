#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
sparta_bin="${repo_root}/benchmarks/sparta/build/src/spa_atlas"

input_file="in.cylinder.smoke"
mpi_ranks=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            input_file="in.cylinder"
            shift
            ;;
        --smoke)
            input_file="in.cylinder.smoke"
            shift
            ;;
        --mpi)
            if [[ $# -lt 2 ]]; then
                echo "Missing rank count after --mpi" >&2
                exit 2
            fi
            mpi_ranks="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--smoke|--full] [--mpi ranks]" >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${sparta_bin}" ]]; then
    echo "SPARTA executable not found: ${sparta_bin}" >&2
    echo "Build it with: cmake --build --preset build-tbb-debug --target atlas_benchmark_sparta_external" >&2
    exit 1
fi

cd "${script_dir}"

if [[ "${mpi_ranks}" -gt 1 ]]; then
    exec mpirun -np "${mpi_ranks}" "${sparta_bin}" -in "${input_file}"
fi

exec "${sparta_bin}" -in "${input_file}"
