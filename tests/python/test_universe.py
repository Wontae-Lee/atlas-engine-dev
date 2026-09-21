import unittest

import numpy as np

from atlas import Float3, Int3, Sphere, Universe


FLOAT_STATES = (
    "temperature",
    "max_relative_speed",
    "max_sigma_g",
    "thermal_energy",
    "number_particle",
    "knudsen_number",
)
VECTOR_STATES = ("bulk_velocity", "field_force", "gravity")
INTEGER_STATES = ("collision_count", "allocated_solver")


class UniverseTests(unittest.TestCase):
    def test_grid_metadata(self):
        universe = Universe(Float3(-1, -2, -3), Float3(1, 2, 3), 1)
        self.assertEqual(universe.lower_corner, Float3(-1, -2, -3))
        self.assertEqual(universe.upper_corner, Float3(1, 2, 3))
        self.assertEqual(universe.grid_size, Int3(3, 5, 7))
        self.assertEqual(universe.cell_count, 105)
        self.assertEqual(universe.cell_size, 1)
        self.assertEqual(universe.cell_volume, 1)
        self.assertEqual(universe.inverse_cell_size, 1)

    def test_from_geometry_uses_its_bound(self):
        universe = Universe.from_geometry(Sphere(Float3(2, 3, 4), 1), 0.5)
        self.assertEqual(universe.lower_corner, Float3(1, 2, 3))
        self.assertEqual(universe.upper_corner, Float3(3, 4, 5))
        self.assertEqual(universe.grid_size, Int3(5, 5, 5))

    def test_all_state_types_support_the_lifecycle(self):
        universe = Universe(Float3(0), Float3(1), 1)
        count = universe.cell_count
        cases = []
        cases.extend((name, np.arange(count, dtype=np.float32), (count,)) for name in FLOAT_STATES)
        vector = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        cases.extend((name, vector, (count, 3)) for name in VECTOR_STATES)
        cases.extend((name, np.arange(count, dtype=np.int32), (count,)) for name in INTEGER_STATES)
        for name, values, shape in cases:
            with self.subTest(state=name):
                self.assertFalse(universe.has_state(name))
                self.assertIsNone(universe.state(name))
                universe.set_state(name, values)
                self.assertTrue(universe.has_state(name))
                self.assertEqual(universe.state(name).shape, shape)
                np.testing.assert_array_equal(universe.state(name), values)
                universe.reset_state(name)
                np.testing.assert_array_equal(universe.state(name), np.zeros(shape, dtype=values.dtype))
                self.assertTrue(universe.remove_state(name))
                self.assertFalse(universe.has_state(name))

    def test_partial_write_and_bounds(self):
        universe = Universe(Float3(0), Float3(1), 1)
        count = universe.cell_count
        universe.set_state("temperature", np.arange(count, dtype=np.float32))
        universe.set_state("temperature", np.array([9], dtype=np.float32), offset=count - 1)
        self.assertEqual(universe.state("temperature")[-1], 9)
        universe.set_state("temperature", np.array([], dtype=np.float32), offset=count)
        with self.assertRaises(ValueError):
            universe.set_state("temperature", np.ones(2, dtype=np.float32), offset=count - 1)

    def test_invalid_grid_parameters_raise(self):
        for lower, upper, cell_size in (
            (Float3(0), Float3(1), 0),
            (Float3(0), Float3(1), -1),
            (Float3(1), Float3(0), 1),
            (Float3(0), Float3(0, 1, 1), 1),
        ):
            with self.subTest(lower=lower, upper=upper, cell_size=cell_size):
                with self.assertRaises(ValueError):
                    Universe(lower, upper, cell_size)
