import gc
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import atlas
from atlas import (
    Atom, Box, Circle, Cylinder, Float3, Fluid, Geometry, Ion, JitteringGenerator,
    Material, MaterialDictionary, MaxwellBoltzmannGenerator, MaxwellSigmaGenerator,
    Molecule, Neutron, Plane, PolygonalPrism, Solid, Sphere, Square, Triangle,
    TriangleMesh, UniformGenerator, Unit, Universe,
)


class ConstructorTests(unittest.TestCase):
    def test_geometry_classes_construct_the_requested_type(self):
        origin, normal = Float3(0), Float3(0, 0, 1)
        for cls, args in (
            (Sphere, (origin, 1)), (Plane, (normal, 0)),
            (Plane, (origin, normal)), (Box, (Float3(-1), Float3(1))),
            (Cylinder, (origin, 1, 2)), (Circle, (origin, normal, 1)),
            (Square, (origin, normal, 1)),
            (Triangle, (origin, Float3(1, 0, 0), Float3(0, 1, 0))),
            (PolygonalPrism, (origin, 6, 1, 2)),
        ):
            with self.subTest(class_name=cls.__name__, args=args):
                self.assertTrue(inspect.isclass(cls))
                value = cls(*args)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, Geometry)
                self.assertTrue(value.is_valid())

    def test_material_classes_construct_the_requested_type(self):
        for cls in (Molecule, Atom, Ion, Neutron, Solid):
            with self.subTest(class_name=cls.__name__):
                value = cls(1) if cls is Solid else cls(1, 0, 0, 0, 1, 273, 1, 1)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, Material)
                self.assertEqual(value.mass(), 1)

    def test_generator_classes_construct_the_requested_type(self):
        for cls in (UniformGenerator, JitteringGenerator, MaxwellSigmaGenerator, MaxwellBoltzmannGenerator):
            with self.subTest(class_name=cls.__name__):
                kwargs = {"species_mass": [1]} if cls is MaxwellBoltzmannGenerator else {}
                value = cls([1], [1], **kwargs)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, atlas.Generator)
                value.bulk_velocity = Float3(1, 2, 3)
                self.assertEqual(value.bulk_velocity, Float3(1, 2, 3))

    def test_policy_classes_construct_the_requested_type(self):
        body = Unit(Sphere(Float3(0), 1))
        for name, base, args in (
            ("VolumeSource", atlas.Source, (body,)),
            ("SurfaceSource", atlas.Source, (body,)),
            ("VolumeSink", atlas.Sink, (body,)),
            ("SurfaceSink", atlas.Sink, (body,)),
            ("TracingSink", atlas.Sink, (body,)),
            ("IsothermalCollider", atlas.Collider, (body,)),
            ("KnudsenCodec", atlas.Codec, ()),
            ("DsmcSolver", atlas.Solver, ()),
        ):
            with self.subTest(class_name=name):
                cls = getattr(atlas, name)
                value = cls(*args)
                self.assertIs(type(value), cls)
                self.assertIsInstance(value, base)

    def test_mesh_constructor_overloads(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)
        face = Triangle(*(Float3(*row) for row in vertices))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            for mesh in (TriangleMesh([face]), TriangleMesh(vertices, indices), TriangleMesh(str(path))):
                with self.subTest(mesh=mesh):
                    self.assertIs(type(mesh), TriangleMesh)
                    self.assertTrue(mesh.is_valid())
        with self.assertRaises(IndexError):
            TriangleMesh(vertices, np.array([[0, 1, 3]], dtype=np.int64))

    def test_state_owner_constructors(self):
        population = Fluid(buffer_size=8, particle_count=2)
        domain = Universe(Float3(0), Float3(1), 0.5)
        self.assertIs(type(population), Fluid)
        self.assertEqual(population.buffer_size, 8)
        self.assertIs(type(domain), Universe)
        self.assertEqual(domain.cell_count, 27)
        self.assertIs(type(MaterialDictionary([Solid(1)])), MaterialDictionary)
        self.assertIs(type(Universe.from_geometry(Sphere(Float3(0), 1), 1)), Universe)

    def test_system_retains_concrete_emitter_classes(self):
        domain = Universe(Float3(0), Float3(1), 0.5)
        domain.set_state("number_particle", np.zeros(domain.cell_count, dtype=np.float32))
        emission = atlas.VolumeSource(Unit(Sphere(Float3(0.5), 0.25)), spacing=0.25)
        velocity = UniformGenerator([1], [1], bulk_velocity=Float3(0.1, 0, 0))
        simulation = atlas.System(Fluid(128), domain, dt=0.01, emitters=[(emission, velocity)])
        del emission, velocity, domain
        gc.collect()
        simulation.update()
        self.assertGreater(simulation.particle_count, 0)
        np.testing.assert_allclose(simulation.velocities()[:, 0], 0.1)

    def test_root_exports_are_classes(self):
        for name in atlas.__all__:
            if name == "__version__":
                continue
            with self.subTest(name=name):
                self.assertTrue(inspect.isclass(getattr(atlas, name)))
                self.assertTrue(name[0].isupper())
