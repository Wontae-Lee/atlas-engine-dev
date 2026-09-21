import gc
import unittest

import numpy as np

from atlas import (
    DsmcSolver,
    Float3,
    Fluid,
    IsothermalCollider,
    KnudsenCodec,
    Sphere,
    System,
    UniformGenerator,
    Unit,
    Universe,
    VolumeSink,
    VolumeSource,
)


def moving_system(dt=0.1):
    positions = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32)
    velocities = np.array([[1, 0, 0], [0, -2, 0.5]], dtype=np.float32)
    return System(
        Fluid.from_arrays(positions, velocities),
        Universe(Float3(-10), Float3(10), 5),
        dt,
    ), positions, velocities


class SystemTests(unittest.TestCase):
    def test_object_hierarchy_matches_core_ownership(self):
        system, _, _ = moving_system()
        self.assertIsInstance(system.fluid, Fluid)
        self.assertIsInstance(system.universe, Universe)
        self.assertIs(system.searcher, system.searcher)
        self.assertIsInstance(system.solvers, list)
        for unwanted in (
            "particle_count",
            "buffer_size",
            "materials",
            "cell_count",
            "lower_corner",
            "upper_corner",
            "grid_size",
            "cell_size",
            "positions",
            "velocities",
            "species",
            "observer",
            "observe",
            "output_directory",
        ):
            with self.subTest(attribute=unwanted):
                self.assertFalse(hasattr(system, unwanted))

    def test_individual_public_phases_are_callable(self):
        system, positions, velocities = moving_system()
        system.initialize_states()
        system.emit()
        system.search()
        self.assertEqual(system.searcher.particle_count, 2)
        system.allocate()
        system.solve()
        system.advect()
        system.remove()
        self.assertEqual(system.step, 0)
        np.testing.assert_allclose(system.fluid.positions(), positions + velocities * system.dt, atol=1e-6)

    def test_update_advances_one_deterministic_step(self):
        system, positions, velocities = moving_system()
        initial = system.step
        system.update()
        self.assertEqual(system.step, initial + 1)
        np.testing.assert_allclose(system.fluid.positions(), positions + velocities * system.dt, atol=1e-6)

    def test_multiple_updates_match_user_workflow(self):
        system, positions, velocities = moving_system(dt=0.05)
        for _ in range(10):
            system.update()
        self.assertEqual(system.step, 10)
        self.assertLessEqual(system.fluid.particle_count, system.fluid.buffer_size)
        self.assertTrue(np.isfinite(system.fluid.positions()).all())
        np.testing.assert_allclose(system.fluid.positions(), positions + velocities * 0.5, atol=1e-5)

    def test_sink_removal_compacts_every_attached_state(self):
        positions = np.array([[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]], dtype=np.float32)
        velocities = np.zeros((2, 3), dtype=np.float32)
        fluid = Fluid.from_arrays(positions, velocities, species=np.array([0, 1], dtype=np.uint64))
        fluid.set_state("temperature", np.array([10, 20], dtype=np.float32))
        system = System(
            fluid,
            Universe(Float3(0), Float3(1), 0.5),
            0.01,
            sinks=[VolumeSink(Unit(Sphere(Float3(0.1), 0.2)))],
        )
        system.update()
        self.assertEqual(system.fluid.particle_count, 1)
        np.testing.assert_allclose(system.fluid.positions(), [[0.8, 0.8, 0.8]], atol=1e-6)
        np.testing.assert_array_equal(system.fluid.species(), [1])
        np.testing.assert_array_equal(system.fluid.state("temperature"), [20])

    def test_component_combinations_construct(self):
        factories = (
            {},
            {"solvers": [DsmcSolver()]},
            {
                "emitters": [(
                    VolumeSource(Unit(Sphere(Float3(0.5), 0.2)), spacing=0.2),
                    UniformGenerator([1], [0]),
                )],
            },
            {"colliders": [IsothermalCollider(Unit(Sphere(Float3(2), 0.2)))]},
            {"sinks": [VolumeSink(Unit(Sphere(Float3(2), 0.2)))]},
            {"solvers": [DsmcSolver()], "codec": KnudsenCodec()},
        )
        for kwargs in factories:
            with self.subTest(arguments=tuple(kwargs)):
                system = System(Fluid(64), Universe(Float3(0), Float3(1), 0.5), 0.01, **kwargs)
                self.assertEqual(system.step, 0)
                self.assertGreater(system.dt, 0)

    def test_consumed_fluid_and_universe_remain_available_through_system(self):
        fluid = Fluid.from_arrays(
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        )
        universe = Universe(Float3(0), Float3(1), 0.5)
        system = System(fluid, universe, 0.1)
        self.assertEqual(system.fluid.particle_count, 1)
        self.assertEqual(system.universe.cell_count, 27)
        del fluid, universe
        gc.collect()
        system.update()
        self.assertEqual(system.fluid.particle_count, 1)
        self.assertEqual(system.universe.cell_count, 27)

    def test_invalid_system_configuration_raises(self):
        with self.assertRaises(RuntimeError):
            System(Fluid(1), Universe(Float3(0), Float3(1), 1), 0)
        with self.assertRaises(RuntimeError):
            System(Fluid(1), Universe(Float3(0), Float3(1), 1), -1)
        with self.assertRaises(ValueError):
            System(
                Fluid(1),
                Universe(Float3(0), Float3(1), 1),
                0.1,
                source=VolumeSource(Unit(Sphere(Float3(0), 1))),
            )
