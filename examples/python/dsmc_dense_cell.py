"""Assemble and run a small DSMC simulation entirely from Python.

Build the `atlas` extension with the bindings enabled, then run this from the
build output directory (or install the wheel), e.g.:

    cmake -S . -B build/tbb -DATLAS_PYTHON=ON -DATLAS_BENCHMARKS=OFF
    cmake --build build/tbb --target atlas_python
    PYTHONPATH=build/tbb/src/python/atlas python examples/python/dsmc_dense_cell.py

The module mirrors the C++ builder API as factory functions: each `atlas.<thing>()`
call returns a ready-to-use object, and `atlas.build_system(...)` wires them into a
runnable System.
"""

import atlas

Vec = atlas.Float3


def main() -> None:
    # One nitrogen (N2) species as a VHS material.
    materials = atlas.material_dictionary(
        [atlas.molecule(
            mass=4.65e-26,
            translational_energy=0.0,
            rotational_energy=0.0,
            vibrational_energy=0.0,
            reference_diameter=4.17e-10,
            reference_temperature=273.0,
            viscosity_index=0.74,
            scattering_parameter=1.0,
        )]
    )

    particle_count = 60
    fluid = atlas.fluid(
        buffer_size=particle_count,
        particle_count=particle_count,
        statistical_weight=1e18,
        materials=materials,
    )
    universe = atlas.universe(Vec(0, 0, 0), Vec(1, 1, 1), cell_size=1.0)
    solver = atlas.dsmc_solver(kernel_type=atlas.DsmcKernelType.variable_hard_sphere)

    system = atlas.build_system(fluid=fluid, universe=universe, dt=1e-4, solver=solver)

    print(f"assembled: cells={system.cell_count} particles={system.particle_count}")
    for _ in range(10):
        system.update()
    print(f"after {system.step} steps: particles={system.particle_count}")

    # Read the live particle state back as numpy arrays (no numpy dependency in the
    # module itself — positions()/velocities() return (N, 3) float32 arrays).
    positions = system.positions()
    speeds = (system.velocities() ** 2).sum(axis=1) ** 0.5
    print(f"positions shape={positions.shape} mean_speed={speeds.mean():.1f} m/s")


if __name__ == "__main__":
    main()
