import gc
import unittest

import numpy as np

from atlas import (
    DsmcSolver,
    Float3,
    Fluid,
    IsothermalCollider,
    MaterialDictionary,
    Molecule,
    Ray,
    System,
    TracingSink,
    Triangle,
    TriangleMesh,
    UniformGenerator,
    Unit,
    Universe,
    VolumeSource,
)


def mesh_unit():
    triangle = Triangle(Float3(0), Float3(1, 0, 0), Float3(0, 1, 0))
    mesh = TriangleMesh([triangle])
    return triangle, mesh, Unit(mesh)


class OwnershipTests(unittest.TestCase):
    def test_unit_keeps_triangle_mesh_alive(self):
        triangle, mesh, unit = mesh_unit()
        del triangle, mesh
        gc.collect()
        hit = unit.trace(Ray(Float3(0.25, 0.25, 1), Float3(0, 0, -1)))
        self.assertTrue(hit.is_intersecting)
        self.assertAlmostEqual(hit.distance, 1)

    def test_policy_objects_keep_triangle_mesh_alive(self):
        for factory, operation in (
            (
                lambda unit: VolumeSource(unit, spacing=0.25, tolerance=0.05),
                lambda value: value.spawn(Fluid(64)),
            ),
            (
                lambda unit: IsothermalCollider(unit, momentum_accommodation_coefficient=0),
                lambda value: value.trace(Float3(0.25, 0.25, 1), Float3(0, 0, -1), 2),
            ),
            (
                lambda unit: TracingSink(unit),
                lambda value: value.despawn(Float3(0.25, 0.25, 1), Float3(0, 0, -1), 2),
            ),
        ):
            triangle, mesh, unit = mesh_unit()
            value = factory(unit)
            del triangle, mesh, unit
            gc.collect()
            with self.subTest(type=type(value).__name__):
                self.assertIsNotNone(operation(value))

    def test_system_keeps_policy_objects_and_mesh_alive(self):
        triangle, mesh, unit = mesh_unit()
        collider = IsothermalCollider(unit, momentum_accommodation_coefficient=0)
        fluid = Fluid.from_arrays(
            np.array([[0.25, 0.25, 0.1]], dtype=np.float32),
            np.array([[0, 0, -1]], dtype=np.float32),
        )
        system = System(
            fluid,
            Universe(Float3(-1), Float3(1), 0.5),
            0.2,
            colliders=[collider],
        )
        del triangle, mesh, unit, collider, fluid
        gc.collect()
        system.update()
        np.testing.assert_allclose(system.fluid.velocities(), [[0, 0, 1]], atol=1e-6)

    def test_system_keeps_source_generator_and_solver_alive(self):
        source = VolumeSource(Unit(TriangleMesh([
            Triangle(Float3(0), Float3(0.5, 0, 0), Float3(0, 0.5, 0)),
        ])), spacing=0.25, tolerance=0.1)
        generator = UniformGenerator([1], [0], bulk_velocity=Float3(0, 0, 1))
        system = System(
            Fluid(64),
            Universe(Float3(-1), Float3(1), 0.5),
            0.01,
            emitters=[(source, generator)],
        )
        del source, generator
        gc.collect()
        system.update()
        self.assertGreater(system.fluid.particle_count, 0)
        self.assertTrue(np.isfinite(system.fluid.velocities()).all())

        solver = DsmcSolver()
        materials = MaterialDictionary([
            Molecule(4.65e-26, 0, 0, 0, 4.17e-10, 273, 0.74, 1),
        ])
        solver_system = System(
            Fluid(4, materials=materials),
            Universe(Float3(0), Float3(1), 1),
            1e-4,
            solvers=[solver],
        )
        del solver, materials
        gc.collect()
        solver_system.update()
        self.assertEqual(solver_system.solver_count, 1)

    def test_repeated_construction_and_collection(self):
        for _ in range(50):
            system = System(
                Fluid.from_arrays(
                    np.zeros((1, 3), dtype=np.float32),
                    np.ones((1, 3), dtype=np.float32),
                ),
                Universe(Float3(0), Float3(1), 1),
                0.01,
            )
            system.update()
            snapshot = system.fluid.positions()
            del system
            gc.collect()
            self.assertTrue(np.isfinite(snapshot).all())

    def test_repeated_mesh_policy_collection(self):
        for _ in range(10):
            triangle, mesh, unit = mesh_unit()
            sink = TracingSink(unit)
            del triangle, mesh, unit
            gc.collect()
            self.assertTrue(
                sink.despawn(Float3(0.25, 0.25, 1), Float3(0, 0, -1), 2))
