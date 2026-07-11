"""Assemble and run a small DSMC simulation entirely from Python.

Build the `atlas` extension with the bindings enabled, then run this from the
build output directory (or install the wheel), e.g.:

    cmake -S . -B build/tbb -DATLAS_PYTHON=ON -DATLAS_BENCHMARKS=OFF
    cmake --build build/tbb --target atlas_python
    PYTHONPATH=build/tbb/src/python/atlas python examples/python/dsmc_dense_cell.py

The module mirrors the C++ builder API as factory functions: each `atlas.<thing>()`
call returns a ready-to-use object, and `atlas.build_system(...)` wires them into a
runnable System. Initial particle state is seeded from numpy arrays and results
are read back the same way.
"""

import atlas
import numpy as np

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

    # Seed a dense cell of fast molecules from numpy so collisions actually fire.
    rng = np.random.default_rng(0)
    particle_count = 200
    positions = rng.uniform(0.1, 0.9, size=(particle_count, 3)).astype(np.float32)
    velocities = (rng.standard_normal((particle_count, 3)) * 300.0).astype(np.float32)

    fluid = atlas.fluid_from_arrays(positions, velocities, statistical_weight=1e18, materials=materials)
    universe = atlas.universe(Vec(0, 0, 0), Vec(1, 1, 1), cell_size=1.0)
    solver = atlas.dsmc_solver(kernel_type=atlas.DsmcKernelType.variable_hard_sphere)

    system = atlas.build_system(fluid=fluid, universe=universe, dt=1e-4, solver=solver)

    mean_speed_before = np.linalg.norm(system.velocities(), axis=1).mean()
    for _ in range(20):
        system.update()
    mean_speed_after = np.linalg.norm(system.velocities(), axis=1).mean()

    print(f"cells={system.cell_count} particles={system.particle_count}")
    print(f"mean speed: {mean_speed_before:.1f} -> {mean_speed_after:.1f} m/s over {system.step} steps")


if __name__ == "__main__":
    main()
