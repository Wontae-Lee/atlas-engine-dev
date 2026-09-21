import inspect
import unittest

import numpy as np

import atlas
from atlas import Float3, Fluid, Geometry, Material, System, Unit, Universe


class ConstructionTests(unittest.TestCase):
    def test_state_owner_construction(self):
        fluid = Fluid(buffer_size=100, particle_count=10, statistical_weight=2.5)
        universe = Universe(Float3(0), Float3(1), 0.5)
        self.assertEqual(fluid.buffer_size, 100)
        self.assertEqual(fluid.particle_count, 10)
        self.assertAlmostEqual(fluid.statistical_weight, 2.5)
        self.assertIsNone(fluid.materials)
        self.assertEqual(universe.cell_count, 27)

    def test_concrete_leaf_classes_keep_their_identity(self):
        geometry_cases = (
            (atlas.Sphere, (Float3(0), 1)),
            (atlas.Plane, (Float3(0, 0, 1), 0)),
            (atlas.Box, (Float3(-1), Float3(1))),
            (atlas.Cylinder, (Float3(0), 1, 2)),
            (atlas.Circle, (Float3(0), Float3(0, 0, 1), 1)),
            (atlas.Square, (Float3(0), Float3(0, 0, 1), 1)),
            (atlas.Triangle, (Float3(0), Float3(1, 0, 0), Float3(0, 1, 0))),
            (atlas.PolygonalPrism, (Float3(0), 6, 1, 2)),
        )
        for cls, args in geometry_cases:
            with self.subTest(cls=cls.__name__):
                value = cls(*args)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, Geometry)
                self.assertTrue(value.is_valid())

        for cls in (atlas.Molecule, atlas.Atom, atlas.Ion, atlas.Neutron, atlas.Solid):
            with self.subTest(cls=cls.__name__):
                args = (1,) if cls is atlas.Solid else (1, 0, 0, 0, 1, 273, 1, 1)
                value = cls(*args)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, Material)

    def test_policy_and_generator_classes_keep_their_identity(self):
        body = Unit(atlas.Sphere(Float3(0), 1))
        cases = (
            (atlas.VolumeSource, atlas.Source, (body,), {}),
            (atlas.SurfaceSource, atlas.Source, (body,), {}),
            (atlas.VolumeSink, atlas.Sink, (body,), {}),
            (atlas.SurfaceSink, atlas.Sink, (body,), {}),
            (atlas.TracingSink, atlas.Sink, (body,), {}),
            (atlas.IsothermalCollider, atlas.Collider, (body,), {}),
            (atlas.KnudsenCodec, atlas.Codec, (), {}),
            (atlas.DsmcSolver, atlas.Solver, (), {}),
        )
        for cls, base, args, kwargs in cases:
            with self.subTest(cls=cls.__name__):
                value = cls(*args, **kwargs)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, base)

        for cls in (
            atlas.UniformGenerator,
            atlas.JitteringGenerator,
            atlas.MaxwellSigmaGenerator,
            atlas.MaxwellBoltzmannGenerator,
        ):
            kwargs = {"species_mass": [1]} if cls is atlas.MaxwellBoltzmannGenerator else {}
            value = cls([1], [0], **kwargs)
            self.assertIs(type(value), cls)
            self.assertIsInstance(value, atlas.Generator)

    def test_system_component_counts(self):
        fluid = Fluid.from_arrays(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        universe = Universe(Float3(0), Float3(1), 0.5)
        solver = atlas.DsmcSolver()
        collider = atlas.IsothermalCollider(Unit(atlas.Sphere(Float3(0.5), 0.1)))
        sink = atlas.VolumeSink(Unit(atlas.Sphere(Float3(2), 0.1)))
        system = System(
            fluid,
            universe,
            0.01,
            solvers=[solver],
            colliders=[collider],
            sinks=[sink],
            codec=atlas.KnudsenCodec(),
        )
        self.assertEqual(system.source_count, 0)
        self.assertEqual(system.solver_count, 1)
        self.assertEqual(system.collider_count, 1)
        self.assertEqual(system.sink_count, 1)
        self.assertEqual(len(system.solvers), 1)
        self.assertIs(system.solvers[0], solver)
        self.assertIsInstance(system.codec, atlas.Codec)
        self.assertTrue(inspect.isclass(type(system)))
