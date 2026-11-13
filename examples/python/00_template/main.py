"""Build and advance a minimal Atlas simulation through the Python binding."""

import argparse

import numpy as np
from atlas import Float3, Fluid, MaterialDictionary, Molecule, System, Universe


def build_system() -> System:
    materials = MaterialDictionary([
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
    ])
    positions = np.array([[-0.5, -0.1, 0.0], [-0.5, 0.1, 0.0]], dtype=np.float32)
    velocities = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    fluid = Fluid.from_arrays(
        positions,
        velocities,
        statistical_weight=1.0,
        materials=materials,
        species=np.zeros(2, dtype=np.uint64),
        buffer_size=1024,
    )
    universe = Universe(Float3(-1.0), Float3(1.0), cell_size=0.25)
    return System(fluid, universe, dt=1.0e-4)


def run(steps: int = 40) -> None:
    system = build_system()
    print(
        f"template: cells={system.universe.cell_count} "
        f"dt={system.dt:.1e}s steps={steps}"
    )
    for _ in range(steps):
        system.update()
    positions = system.fluid.positions()
    print(
        f"completed: steps={system.step} particles={system.fluid.particle_count} "
        f"first_x={positions[0, 0]:.6f}"
    )


def nonnegative_steps(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("steps must be a nonnegative decimal integer")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal Atlas Python template.")
    parser.add_argument(
        "steps", type=nonnegative_steps, nargs="?", default=40,
        help="number of simulation updates (default: 40; 0 only builds and reports)",
    )
    run(parser.parse_args().steps)


if __name__ == "__main__":
    main()
