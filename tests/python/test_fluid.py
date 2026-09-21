import unittest

import numpy as np

from atlas import Fluid, MaterialDictionary, Solid


OPTIONAL_STATES = (
    "temperature",
    "translational_energy",
    "rotational_energy",
    "vibrational_energy",
)


class FluidTests(unittest.TestCase):
    def test_from_arrays_handles_empty_single_and_spare_capacity(self):
        for count, capacity in ((0, None), (1, 1), (3, 3), (3, 7)):
            positions = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
            velocities = -positions
            kwargs = {} if capacity is None else {"buffer_size": capacity}
            with self.subTest(count=count, capacity=capacity):
                fluid = Fluid.from_arrays(positions, velocities, **kwargs)
                self.assertEqual(fluid.particle_count, count)
                self.assertEqual(fluid.buffer_size, capacity if capacity is not None else max(count, 1))
                np.testing.assert_array_equal(fluid.positions(), positions)
                np.testing.assert_array_equal(fluid.velocities(), velocities)

    def test_mandatory_states_and_shorthand_match(self):
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        velocities = -positions
        species = np.array([1, 0], dtype=np.uint64)
        fluid = Fluid.from_arrays(positions, velocities, species=species)
        for name in ("position", "velocity", "species"):
            self.assertTrue(fluid.has_state(name))
        np.testing.assert_array_equal(fluid.state("position"), fluid.positions())
        np.testing.assert_array_equal(fluid.state("velocity"), fluid.velocities())
        np.testing.assert_array_equal(fluid.state("species"), fluid.species())
        self.assertEqual(fluid.positions().shape, (2, 3))
        self.assertEqual(fluid.positions().dtype, np.float32)
        self.assertEqual(fluid.species().shape, (2,))
        self.assertEqual(fluid.species().dtype, np.uint64)

    def test_optional_state_lifecycle(self):
        fluid = Fluid(4, particle_count=3)
        values = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        for name in OPTIONAL_STATES:
            with self.subTest(state=name):
                self.assertFalse(fluid.has_state(name))
                self.assertIsNone(fluid.state(name))
                fluid.set_state(name, values)
                self.assertTrue(fluid.has_state(name))
                np.testing.assert_array_equal(fluid.state(name), values)
                fluid.reset_state(name)
                np.testing.assert_array_equal(fluid.state(name), np.zeros(3, dtype=np.float32))
                self.assertTrue(fluid.remove_state(name))
                self.assertFalse(fluid.has_state(name))
                self.assertIsNone(fluid.state(name))

    def test_partial_writes_and_capacity_boundaries(self):
        fluid = Fluid(5, particle_count=5)
        fluid.set_state("temperature", np.arange(5, dtype=np.float32))
        fluid.set_state("temperature", np.array([20, 30], dtype=np.float32), offset=2)
        np.testing.assert_array_equal(fluid.state("temperature"), [0, 1, 20, 30, 4])
        fluid.set_state("temperature", np.array([], dtype=np.float32), offset=5)
        with self.assertRaises(ValueError):
            fluid.set_state("temperature", np.array([1], dtype=np.float32), offset=5)
        with self.assertRaises((TypeError, OverflowError)):
            fluid.set_state("temperature", np.array([], dtype=np.float32), offset=-1)
        with self.assertRaises(ValueError):
            fluid.set_state("temperature", np.ones(6, dtype=np.float32))

    def test_particle_count_controls_live_reads(self):
        fluid = Fluid(4)
        fluid.set_state("position", np.arange(12, dtype=np.float32).reshape(4, 3))
        for count in (0, 1, 4):
            fluid.set_particle_count(count)
            self.assertEqual(fluid.particle_count, count)
            self.assertEqual(fluid.positions().shape, (count, 3))
            self.assertEqual(fluid.state("position", full=True).shape, (4, 3))
        with self.assertRaises(IndexError):
            fluid.particle_count = 5

    def test_compaction_keeps_every_state_aligned(self):
        positions = np.arange(15, dtype=np.float32).reshape(5, 3)
        velocities = positions + 100
        species = np.arange(5, dtype=np.uint64)
        fluid = Fluid.from_arrays(positions, velocities, species=species)
        fluid.set_state("temperature", np.arange(5, dtype=np.float32) + 200)
        fluid.set_active(np.array([1, 0, 1, 0, 1], dtype=np.int32))
        self.assertEqual(fluid.compact(), 3)
        self.assertEqual(fluid.particle_count, 3)
        np.testing.assert_array_equal(fluid.positions(), positions[[0, 2, 4]])
        np.testing.assert_array_equal(fluid.velocities(), velocities[[0, 2, 4]])
        np.testing.assert_array_equal(fluid.species(), species[[0, 2, 4]])
        np.testing.assert_array_equal(fluid.state("temperature"), [200, 202, 204])
        np.testing.assert_array_equal(fluid.active(), [1, 1, 1])

    def test_active_writes_validate_values_and_capacity(self):
        fluid = Fluid(3, particle_count=3)
        fluid.set_active(np.array([1, 0], dtype=np.int32), offset=1)
        np.testing.assert_array_equal(fluid.active(full=True), [0, 1, 0])
        fluid.set_active(np.array([], dtype=np.int32), offset=3)
        with self.assertRaises(ValueError):
            fluid.set_active(np.array([1], dtype=np.int32), offset=3)
        with self.assertRaises(ValueError):
            fluid.set_active(np.array([2], dtype=np.int32))

    def test_species_are_checked_against_material_dictionary(self):
        materials = MaterialDictionary([Solid(1), Solid(2)])
        fluid = Fluid(3, particle_count=2, materials=materials)
        fluid.set_state("species", np.array([0, 1], dtype=np.uint64))
        with self.assertRaises(ValueError):
            fluid.set_state("species", np.array([2], dtype=np.uint64))

    def test_invalid_from_arrays_inputs_raise(self):
        positions = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            Fluid.from_arrays(positions, np.zeros((1, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            Fluid.from_arrays(positions, positions, species=np.zeros(1, dtype=np.uint64))
        with self.assertRaises(RuntimeError):
            Fluid.from_arrays(positions, positions, buffer_size=1)
        with self.assertRaises(ValueError):
            Fluid.from_arrays(
                positions,
                positions,
                materials=MaterialDictionary([Solid(1)]),
                species=np.array([0, 1], dtype=np.uint64),
            )
