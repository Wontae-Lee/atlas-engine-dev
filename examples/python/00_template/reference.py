"""Reference construction surface for a new Atlas Python simulation."""

import numpy as np

from atlas import (
    Atom,
    Box,
    Circle,
    Cylinder,
    DiffuseSampling,
    DsmcKernelType,
    DsmcSolver,
    Float3,
    Fluid,
    Ion,
    IsothermalCollider,
    JitteringGenerator,
    KnudsenCodec,
    MaterialDictionary,
    MaxwellBoltzmannGenerator,
    MaxwellSigmaGenerator,
    Molecule,
    Neutron,
    Plane,
    PolygonalPrism,
    Quaternion,
    Solid,
    Sphere,
    Square,
    SurfaceSink,
    SurfaceSource,
    Sync,
    System,
    TracingSink,
    Triangle,
    TriangleMesh,
    UniformGenerator,
    Unit,
    Universe,
    VolumeSink,
    VolumeSource,
)


def material_examples() -> MaterialDictionary:
    """Construct one instance of every material type exposed to Python."""
    gas = dict(
        mass=4.65e-26,
        translational_energy=0.0,
        rotational_energy=0.0,
        vibrational_energy=0.0,
        reference_diameter=4.17e-10,
        reference_temperature=273.0,
        viscosity_index=0.74,
        scattering_parameter=1.0,
    )
    return MaterialDictionary(
        [
            Molecule(**gas),
            Atom(**gas),
            Ion(**gas),
            Neutron(**gas),
            Solid(mass=1.0),
        ]
    )


def geometry_examples():
    """Construct representative geometry leaves keyed by their public names."""
    triangle = Triangle(Float3(0, 0, 0), Float3(1, 0, 0), Float3(0, 1, 0))
    mesh = TriangleMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    return {
        "box": Box(Float3(-1), Float3(1)),
        "circle": Circle(Float3(0), Float3(0, 0, 1), 1.0),
        "cylinder": Cylinder(Float3(0), 1.0, 2.0, open=False),
        "plane": Plane(Float3(0, 1, 0), 0.0),
        "sphere": Sphere(Float3(0), 1.0),
        "square": Square(Float3(0), Float3(0, 0, 1), 1.0),
        "triangle": triangle,
        "triangle_mesh": mesh,
        "polygonal_prism": PolygonalPrism(Float3(0), 6, 1.0, 2.0),
    }


def build_system() -> System:
    """Construct a reference System covering the main Python binding surfaces."""
    materials = material_examples()
    positions = np.array([[-0.5, -0.1, 0.0], [-0.5, 0.1, 0.0]], dtype=np.float32)
    velocities = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    species = np.zeros(2, dtype=np.uint64)
    fluid = Fluid.from_arrays(
        positions,
        velocities,
        statistical_weight=1.0,
        materials=materials,
        species=species,
        buffer_size=1024,
    )
    # Optional Fluid states are created explicitly and remain owned by Fluid.
    for name in (
        "temperature",
        "translational_energy",
        "rotational_energy",
        "vibrational_energy",
    ):
        fluid.set_state(name, np.zeros(2, dtype=np.float32))

    universe = Universe(Float3(-1), Float3(1), 0.25)
    universe.set_state("temperature", np.full(universe.cell_count, 300.0, dtype=np.float32))
    universe.set_state(
        "bulk_velocity", np.zeros((universe.cell_count, 3), dtype=np.float32)
    )
    # Universe.from_geometry(Box(Float3(-1), Float3(1)), cell_size=0.25) is the bound-derived form.

    # Units combine geometry with rigid pose and optional kinematics.
    geometries = geometry_examples()
    moving_unit = Unit(
        geometries["cylinder"],
        sync=Sync(Float3(0), Quaternion()),
        velocity=Float3(0.1, 0, 0),
        acceleration=Float3(0),
        angular_velocity=Float3(0, 0, 0.1),
        angular_acceleration=Float3(0),
    )

    volume_source = VolumeSource(Unit(geometries["box"]), spacing=0.1, tolerance=0.0)
    surface_source = SurfaceSource(Unit(geometries["sphere"]), spacing=0.1, tolerance=0.01)
    generators = {
        "uniform": UniformGenerator([1.0], [0.0], min_value=-1.0, max_value=1.0),
        "jittering": JitteringGenerator([1.0], [0.0], base_value=0.0, jitter_radius=0.1),
        "maxwell_sigma": MaxwellSigmaGenerator([1.0], [0.0], sigma=1.0),
        "maxwell_boltzmann": MaxwellBoltzmannGenerator(
            [1.0], [0.0], materials=materials, temperature=300.0
        ),
    }

    collider = IsothermalCollider(
        moving_unit,
        momentum_accommodation_coefficient=1.0,
        restitution=1.0,
        diffuse_sampling=DiffuseSampling.uniform,
    )
    sinks = [
        SurfaceSink(Unit(geometries["plane"]), tolerance=0.0),
        VolumeSink(Unit(geometries["sphere"]), tolerance=0.0),
        TracingSink(Unit(geometries["square"])),
    ]
    solver = DsmcSolver(
        DsmcKernelType.variable_hard_sphere,
        majorant_sample_pairs=8,
        majorant_exhaustive_limit=5,
    )
    codec = KnudsenCodec(
        representative_characteristic_length=1.0,
        representative_collision_cross_sectional_area=1.0,
        representative_statistical_weight=1.0,
        representative_cell_volume=1.0,
    )

    return System(
        fluid,
        universe,
        1.0e-4,
        solvers=[solver],
        emitters=[
            (volume_source, generators["maxwell_boltzmann"]),
            (surface_source, generators["uniform"]),
        ],
        colliders=[collider],
        sinks=sinks,
        codec=codec,
    )
