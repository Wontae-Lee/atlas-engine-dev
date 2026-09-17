import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas import (
    BVH, Codec, Collider, DefaultRandomEngine, DsmcSolver, Float3, Fluid,
    Geometry, Int3, IsothermalCollider, KnudsenCodec, LBVH, MaterialDictionary,
    MaterialType, Molecule, Observer, Ray, SAHBVH, Solver, SpatialHashingSearcher,
    Sphere, Sync, System, Triangle, TriangleMesh, UniformGenerator, Unit,
    Universe, VolumeSink, VolumeSource,
)
from atlas.sampling import sample_random_unit_vector
from atlas.serialization import (
    load_fluid_binary, restore_fluid, restore_universe,
    save_fluid_binary, save_universe_binary,
)


def make_fluid():
    return Fluid.from_arrays(
        np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], dtype=np.float32),
        np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32),
        buffer_size=4,
    )


def make_universe():
    domain = Universe(Float3(0), Float3(1), 0.5)
    domain.set_state("number_particle", np.zeros(domain.cell_count, dtype=np.float32))
    return domain


class RuntimeTests(unittest.TestCase):
    def test_geometry_queries_and_pose(self):
        shape = Sphere(Float3(0), 1)
        self.assertIsInstance(shape, Geometry)
        self.assertAlmostEqual(shape.signed_distance(Float3(2, 0, 0)), 1)
        body = Unit(shape, Sync(translation=Float3(3, 0, 0)))
        bound = body.world_bound()
        self.assertTrue(bound.contains(Float3(3, 0, 0)))
        self.assertFalse(bound.contains(Float3(0)))
        hit = shape.trace(Ray(Float3(-2, 0, 0), Float3(1, 0, 0)))
        self.assertTrue(hit.is_intersecting)
        self.assertAlmostEqual(hit.distance, 1)

    def test_mesh_owner_survives_intermediate_handles(self):
        face = Triangle(Float3(0), Float3(1, 0, 0), Float3(0, 1, 0))
        mesh = TriangleMesh([face])
        body = Unit(mesh)
        retained = body.geometry
        del face, mesh, body
        gc.collect()
        hit = retained.trace(Ray(Float3(0.25, 0.25, 1), Float3(0, 0, -1)))
        self.assertTrue(hit.is_intersecting)
        self.assertAlmostEqual(hit.distance, 1)

    def test_bvh_builds_triangle_bounds(self):
        face = Triangle(Float3(0), Float3(1, 0, 0), Float3(0, 1, 0))
        for cls in (LBVH, SAHBVH):
            with self.subTest(type=cls.__name__):
                hierarchy = cls([face])
                self.assertIsInstance(hierarchy, BVH)
                self.assertEqual(hierarchy.indices(), [0])
                self.assertTrue(hierarchy.bounds()[0].contains(Float3(0.25, 0.25, 0)))

    def test_state_upload_and_readback_are_independent(self):
        population = make_fluid()
        self.assertEqual(population.particle_count, 2)
        self.assertEqual(population.buffer_size, 4)
        copied = population.positions()
        copied[:] = 9
        self.assertLess(population.positions().max(), 1)
        population.set_state("position", np.array([[0.5, 0.5, 0.5]], dtype=np.float32), offset=1)
        np.testing.assert_array_equal(population.positions()[1], [0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            Fluid.from_arrays(np.zeros((2, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32))

    def test_spatial_hash_classification(self):
        population, domain = make_fluid(), make_universe()
        index = SpatialHashingSearcher(domain)
        index.classify(population, domain)
        self.assertEqual(index.particle_count, 2)
        self.assertEqual(domain.grid_size, Int3(3, 3, 3))
        np.testing.assert_array_equal(np.sort(index.cell_key()), [0, 13])
        np.testing.assert_array_equal(np.sort(index.indices()), [0, 1])
        self.assertEqual(domain.state("number_particle").sum(), 2)

    def test_sources_generators_and_boundaries(self):
        body = Unit(Sphere(Float3(0.5), 0.4))
        emitter = VolumeSource(body, spacing=0.2)
        population = Fluid(buffer_size=256)
        count = emitter.spawn(population)
        self.assertGreater(count, 0)
        velocity = UniformGenerator([1], [1], bulk_velocity=Float3(2, 0, 0))
        self.assertEqual(velocity.generate(population, 0, count), count)
        population.particle_count = count
        np.testing.assert_allclose(population.velocities(), np.tile([2, 0, 0], (count, 1)))
        self.assertTrue(VolumeSink(body).despawn(Float3(0.5), Float3(0), 0.1))
        self.assertIsInstance(IsothermalCollider(body), Collider)
        self.assertIsInstance(KnudsenCodec(), Codec)
        self.assertIsInstance(DsmcSolver(), Solver)

    def test_material_dictionary_preserves_species(self):
        species = Molecule(4.65e-26, 0, 0, 0, 4.17e-10, 273, 0.74, 1)
        table = MaterialDictionary([species])
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0].type, MaterialType.molecule)
        self.assertAlmostEqual(table[0].mass() / species.mass(), 1)
        with self.assertRaises(IndexError):
            _ = table[1]

    def test_seeded_sampling_is_reproducible(self):
        first, second = DefaultRandomEngine(42), DefaultRandomEngine(42)
        self.assertEqual([first() for _ in range(8)], [second() for _ in range(8)])
        direction = sample_random_unit_vector(first)
        self.assertAlmostEqual(direction.length(), 1, places=5)

    def test_binary_snapshot_round_trip(self):
        population, domain = make_fluid(), make_universe()
        with tempfile.TemporaryDirectory() as directory:
            particles_path, grid_path = Path(directory) / "fluid.bin", Path(directory) / "universe.bin"
            save_fluid_binary(population, particles_path)
            save_universe_binary(domain, grid_path)
            restored = restore_fluid(particles_path)
            restored_grid = restore_universe(grid_path)
            np.testing.assert_array_equal(restored.positions(), population.positions())
            self.assertEqual(restored.buffer_size, population.buffer_size)
            self.assertEqual(restored_grid.cell_count, domain.cell_count)
            decoded = load_fluid_binary(particles_path)
            self.assertEqual(decoded["particle_count"], 2)

    def test_system_advances_owned_state(self):
        population, domain = make_fluid(), make_universe()
        before = population.positions()
        velocities = population.velocities()
        with tempfile.TemporaryDirectory() as directory:
            recorder = Observer(1, directory)
            simulation = System(population, domain, dt=0.01, observer=recorder)
            del population, domain, recorder
            gc.collect()
            simulation.update()
            self.assertEqual(simulation.step, 1)
            self.assertEqual(simulation.particle_count, 2)
            np.testing.assert_allclose(simulation.positions(), before + velocities * 0.01, atol=1e-6)
