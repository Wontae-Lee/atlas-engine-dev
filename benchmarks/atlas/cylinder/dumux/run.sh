#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
dumux_bin="${repo_root}/benchmarks/dumux/build/test/freeflow/navierstokes/unstructured/test_ff_navierstokes_dfg_benchmark_stationary_pq1bubble_diamond"

mesh_file="cylinder_channel.msh"
problem_name="atlas_dumux_cylinder_smoke"
check_indicators="false"
mpi_ranks=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            mesh_file="cylinder_channel_quad.msh"
            problem_name="atlas_dumux_cylinder_full"
            check_indicators="true"
            shift
            ;;
        --smoke)
            mesh_file="cylinder_channel.msh"
            problem_name="atlas_dumux_cylinder_smoke"
            check_indicators="false"
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

if [[ ! -x "${dumux_bin}" ]]; then
    echo "DuMux executable not found: ${dumux_bin}" >&2
    echo "Build it with: cmake --build --preset build-tbb-debug --target atlas_benchmark_dumux_external" >&2
    exit 1
fi

cd "${script_dir}"

args=(
    "params.input"
    "-Problem.Name" "${problem_name}"
    "-Problem.EnableInertiaTerms" "true"
    "-Problem.CheckIndicators" "${check_indicators}"
    "-Grid.File" "${mesh_file}"
)

if [[ "${mpi_ranks}" -gt 1 ]]; then
    exec mpirun -np "${mpi_ranks}" "${dumux_bin}" "${args[@]}"
fi

exec "${dumux_bin}" "${args[@]}"
