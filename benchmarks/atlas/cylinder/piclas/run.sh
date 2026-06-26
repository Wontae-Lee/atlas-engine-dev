#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
piclas_bin="${repo_root}/benchmarks/piclas/build/bin/piclas"

parameter_file="parameter.smoke.ini"
species_file="DSMC.ini"
mpi_ranks=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            parameter_file="parameter.ini"
            shift
            ;;
        --smoke)
            parameter_file="parameter.smoke.ini"
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

if [[ ! -x "${piclas_bin}" ]]; then
    echo "PICLas executable not found: ${piclas_bin}" >&2
    echo "Build it with: cmake --build --preset build-tbb-debug --target atlas_benchmark_piclas_external" >&2
    exit 1
fi

cd "${script_dir}"

if [[ "${mpi_ranks}" -gt 1 ]]; then
    exec mpirun -np "${mpi_ranks}" "${piclas_bin}" "${parameter_file}" "${species_file}"
fi

exec "${piclas_bin}" "${parameter_file}" "${species_file}"
