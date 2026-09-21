import gc
import unittest

import numpy as np

from atlas import (
    DsmcKernelType, DsmcSolver, Float3, Fluid, IsothermalCollider,
    MaterialDictionary, MaterialType, Molecule, Solid,
    SpatialHashingSearcher, System, Triangle, TriangleMesh, Unit,
    Universe, VolumeSink,
)
from atlas.math import tol


class BackendParityTests(unittest.TestCase):
    def test_mesh_collider_updates_and_reflects_particles(self):
        for wall_velocity in (None, Float3(0.1, 0, 0)):
            with self.subTest(wall_velocity=wall_velocity):
                mesh = TriangleMesh([
                    Triangle(Float3(0), Float3(1, 0, 0), Float3(0, 1, 0)),
                ])
                wall = IsothermalCollider(
                    Unit(mesh, velocity=wall_velocity),
                    momentum_accommodation_coefficient=0,
                )
                population = Fluid.from_arrays(
                    np.array([[0.25, 0.25, 0.125]], dtype=np.float32),
                    np.array([[0, 0, -1]], dtype=np.float32),
                )
                simulation = System(
                    population, Universe(Float3(-1), Float3(1), 0.5),
                    dt=0.25, colliders=[wall],
                )
                del wall, mesh, population
                gc.collect()
                simulation.update()
                np.testing.assert_allclose(simulation.fluid.velocities(), [[0, 0, 1]], atol=1e-6)
                np.testing.assert_allclose(simulation.fluid.positions(), [[0.25, 0.25, tol]], atol=1e-6)
                simulation.update()
                np.testing.assert_allclose(simulation.fluid.positions(), [[0.25, 0.25, 0.25 + tol]], atol=1e-6)

    def test_mesh_sink_removes_and_compacts_particles(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        indices = np.array([
            [0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3],
        ], dtype=np.int64)
        boundary = VolumeSink(Unit(TriangleMesh(vertices, indices)))
        population = Fluid.from_arrays(
            np.array([[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]], dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        )
        simulation = System(
            population, Universe(Float3(0), Float3(1), 0.5),
            dt=0.01, sinks=[boundary],
        )
        del boundary, population
        gc.collect()
        simulation.update()
        self.assertEqual(simulation.fluid.particle_count, 1)
        np.testing.assert_allclose(simulation.fluid.positions(), [[0.8, 0.8, 0.8]], atol=1e-6)

    def test_material_dictionary_uploads_replacements(self):
        table = MaterialDictionary([Solid(1), Solid(2)])
        previous = table.materials()
        table[-1] = Solid(3)
        self.assertEqual(table[1].type, MaterialType.solid)
        self.assertEqual(table[1].mass(), 3)
        self.assertEqual(previous[1].mass(), 2)
        table.set_materials([Solid(4)])
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0].mass(), 4)
        with self.assertRaises(IndexError):
            table[1] = Solid(5)

    def test_dsmc_collision_models_conserve_momentum_and_energy(self):
        positions = np.array([
            [0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.4, 0.2, 0.2],
        ], dtype=np.float32)
        velocities = np.array([
            [300, 0, 0], [-200, 100, 0], [0, -100, 250],
        ], dtype=np.float32)
        before = velocities.astype(np.float64)
        expected_momentum = before.sum(axis=0)
        expected_energy = np.square(before).sum()
        momentum_tolerance = np.linalg.norm(before, axis=1).sum() * 1e-4
        for kernel in (
            DsmcKernelType.hard_sphere,
            DsmcKernelType.variable_hard_sphere,
            DsmcKernelType.variable_soft_sphere,
        ):
            with self.subTest(kernel=kernel):
                materials = MaterialDictionary([
                    Molecule(4.65e-26, 0, 0, 0, 4.17e-10, 273, 0.74, 1),
                ])
                population = Fluid.from_arrays(
                    positions, velocities, statistical_weight=1e22,
                    materials=materials,
                )
                domain = Universe(Float3(0), Float3(0.9), 1)
                for state in ("number_particle", "max_relative_speed", "max_sigma_g"):
                    domain.set_state(state, np.zeros(domain.cell_count, dtype=np.float32))
                domain.set_state("collision_count", np.zeros(domain.cell_count, dtype=np.int32))
                index = SpatialHashingSearcher(domain)
                index.classify(population, domain)
                solver = DsmcSolver(kernel_type=kernel)
                index.solve(solver, population, domain, index=0, dt=1e-4)
                self.assertGreater(domain.state("collision_count").sum(), 0)
                after = population.velocities().astype(np.float64)
                self.assertTrue(np.isfinite(after).all())
                self.assertFalse(np.array_equal(after, before))
                np.testing.assert_allclose(after.sum(axis=0), expected_momentum, rtol=0, atol=momentum_tolerance)
                self.assertAlmostEqual(np.square(after).sum() / expected_energy, 1, delta=1e-4)
