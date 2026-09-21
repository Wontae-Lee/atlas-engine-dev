"""Run a compact DSMC nitrogen flow over a cylinder.

Install Atlas from the checkout, then run:

    python examples/python/cylinder.py

Pass a step count as the first argument to run a longer case.
"""

import sys
from typing import Tuple

import numpy as np

from atlas import (
    Box,
    Cylinder,
    DsmcSolver,
    Float3,
    Fluid,
    IsothermalCollider,
    MaterialDictionary,
    MaxwellBoltzmannGenerator,
    Molecule,
    System,
    Unit,
    Universe,
    VolumeSink,
    VolumeSource,
)


def box_unit(
    lower: Tuple[float, float, float],
    upper: Tuple[float, float, float],
) -> Unit:
    return Unit(Box(Float3(*lower), Float3(*upper)))


def main() -> None:
    domain_lower = (-1.0, -0.75, -0.5)
    domain_upper = (1.5, 0.75, 0.5)
    cell_size = 0.1
    dt = 5.0e-5
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    freestream_speed = 500.0
    temperature = 300.0
    cylinder_radius = 0.2

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

    fluid = Fluid(
        buffer_size=20_000,
        statistical_weight=3.236e16,
        materials=materials,
    )
    universe = Universe(Float3(*domain_lower), Float3(*domain_upper), cell_size)

    inflow = VolumeSource(
        box_unit((-0.8, -0.5, -0.35), (-0.7, 0.5, 0.35)),
        spacing=cell_size,
    )
    freestream = MaxwellBoltzmannGenerator(
        species_ratios=[1.0],
        species_numbers=[0.0],
        materials=materials,
        temperature=temperature,
        bulk_velocity=Float3(freestream_speed, 0.0, 0.0),
        seed=20260921,
    )

    cylinder = IsothermalCollider(
        Unit(Cylinder(Float3(0.0), cylinder_radius, 1.0, open=True)),
        momentum_accommodation_coefficient=1.0,
        restitution=1.0,
    )

    sinks = [
        VolumeSink(box_unit((-1.0, -0.75, -0.5), (-0.9, 0.75, 0.5))),
        VolumeSink(box_unit((1.35, -0.75, -0.5), (1.5, 0.75, 0.5))),
        VolumeSink(box_unit((-1.0, -0.75, -0.5), (1.5, -0.6, 0.5))),
        VolumeSink(box_unit((-1.0, 0.6, -0.5), (1.5, 0.75, 0.5))),
        VolumeSink(box_unit((-1.0, -0.75, -0.5), (1.5, 0.75, -0.4))),
        VolumeSink(box_unit((-1.0, -0.75, 0.4), (1.5, 0.75, 0.5))),
    ]

    system = System(
        fluid,
        universe,
        dt,
        solver=DsmcSolver(),
        emitters=[(inflow, freestream)],
        colliders=[cylinder],
        sinks=sinks,
    )

    print(
        f"cylinder flow: cells={system.universe.cell_count} "
        f"dt={system.dt:.1e}s steps={steps}"
    )

    for _ in range(steps):
        system.update()
        if system.step % 10 == 0 or system.step == steps:
            print(f"step={system.step:3d} particles={system.fluid.particle_count:5d}")

    positions = system.fluid.positions()
    velocities = system.fluid.velocities()
    downstream = int(np.count_nonzero(positions[:, 0] > cylinder_radius))
    mean_streamwise_velocity = float(velocities[:, 0].mean()) if velocities.size else 0.0

    print(
        f"completed: steps={system.step} particles={system.fluid.particle_count} "
        f"downstream={downstream} mean_u={mean_streamwise_velocity:.1f}m/s"
    )


if __name__ == "__main__":
    main()
