"""Assemble and run a small DSMC simulation through the Atlas Python bindings.

Install the package from a checkout, then run the example:

    python -m pip install -e .
    python examples/python/dsmc_dense_cell.py

The public modules mirror the C++ module layout and expose types and functions
implemented by the compiled ``atlas._core`` extension.
"""

import numpy as np

from atlas import DsmcSolver, Float3, Fluid, MaterialDictionary, Molecule, System, Universe


def main() -> None:
    # One nitrogen (N2) species as a VHS material.
    materials = MaterialDictionary(
        [
            Molecule(
                mass=4.65e-26,
                translational_energy=0.0,
                rotational_energy=0.0,
                vibrational_energy=0.0,
                reference_diameter=4.17e-10,
                reference_temperature=273.0,
                viscosity_index=0.74,
                scattering_parameter=1.0,
            )
        ]
    )

    # Seed a dense cell of fast molecules from numpy so collisions actually fire.
    rng = np.random.default_rng(0)
    particle_count = 200
    positions = rng.uniform(0.1, 0.9, size=(particle_count, 3)).astype(np.float32)
    velocities = (rng.standard_normal((particle_count, 3)) * 300.0).astype(np.float32)

    fluid = Fluid.from_arrays(
        positions,
        velocities,
        statistical_weight=1e18,
        materials=materials,
    )
    domain = Universe(Float3(0, 0, 0), Float3(1, 1, 1), cell_size=1.0)
    solver = DsmcSolver()

    system = System(fluid=fluid, universe=domain, dt=1e-4, solver=solver)

    mean_speed_before = np.linalg.norm(system.fluid.velocities(), axis=1).mean()
    for _ in range(20):
        system.update()
    mean_speed_after = np.linalg.norm(system.fluid.velocities(), axis=1).mean()

    print(f"cells={system.universe.cell_count} particles={system.fluid.particle_count}")
    print(f"mean speed: {mean_speed_before:.1f} -> {mean_speed_after:.1f} m/s over {system.step} steps")


if __name__ == "__main__":
    main()
