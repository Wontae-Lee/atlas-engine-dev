import gc
import unittest

import numpy as np

from atlas import Float3, Fluid, Universe


class NumPyTransferTests(unittest.TestCase):
    def test_float3_component_order_round_trips(self):
        values = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-7.0, 8.5, 9.25]],
            dtype=np.float32,
        )
        fluid = Fluid(3, particle_count=3)
        fluid.set_state("position", values)
        np.testing.assert_array_equal(fluid.positions(), values)
        replacement = values[:, ::-1].copy()
        fluid.set_state("velocity", replacement)
        np.testing.assert_array_equal(fluid.velocities(), replacement)

    def test_scalar_and_integer_round_trips_preserve_dtype(self):
        fluid = Fluid(3, particle_count=3)
        temperatures = np.array([1.25, -2.5, 3.75], dtype=np.float32)
        species = np.array([0, 4, 2], dtype=np.uint64)
        fluid.set_state("temperature", temperatures)
        fluid.set_state("species", species)
        self.assertEqual(fluid.state("temperature").dtype, np.float32)
        self.assertEqual(fluid.species().dtype, np.uint64)
        np.testing.assert_array_equal(fluid.state("temperature"), temperatures)
        np.testing.assert_array_equal(fluid.species(), species)

        universe = Universe(Float3(0), Float3(1), 1)
        integers = np.arange(universe.cell_count, dtype=np.int32)
        universe.set_state("collision_count", integers)
        self.assertEqual(universe.state("collision_count").dtype, np.int32)
        np.testing.assert_array_equal(universe.state("collision_count"), integers)

    def test_reads_are_owned_snapshots(self):
        fluid = Fluid.from_arrays(
            np.array([[1, 2, 3]], dtype=np.float32),
            np.array([[4, 5, 6]], dtype=np.float32),
        )
        positions = fluid.positions()
        positions[0] = 99
        np.testing.assert_array_equal(fluid.positions(), [[1, 2, 3]])
        del fluid
        gc.collect()
        np.testing.assert_array_equal(positions, [[99, 99, 99]])

        universe = Universe(Float3(0), Float3(1), 1)
        values = np.arange(universe.cell_count, dtype=np.float32)
        universe.set_state("temperature", values)
        snapshot = universe.state("temperature")
        del universe
        gc.collect()
        np.testing.assert_array_equal(snapshot, values)

    def test_old_snapshot_survives_update_and_compaction(self):
        fluid = Fluid.from_arrays(
            np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        )
        old = fluid.positions()
        fluid.set_active(np.array([0, 1], dtype=np.int32))
        fluid.compact()
        np.testing.assert_array_equal(old, [[0, 0, 0], [1, 1, 1]])

    def test_invalid_array_shapes_and_dtypes_raise(self):
        fluid = Fluid(3, particle_count=3)
        invalid_shapes = (
            np.zeros(3, dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
        )
        for values in invalid_shapes:
            with self.subTest(shape=values.shape, dtype=values.dtype):
                with self.assertRaises(RuntimeError):
                    fluid.set_state("position", values)
        with self.assertRaises(RuntimeError):
            fluid.set_state("temperature", np.zeros((3, 1), dtype=np.float32))
        converted_integer_dtype = np.array([2, 1, 0], dtype=np.int32)
        fluid.set_state("species", converted_integer_dtype)
        np.testing.assert_array_equal(fluid.species(), converted_integer_dtype.astype(np.uint64))

        converted_dtype = np.arange(9, dtype=np.float64).reshape(3, 3)
        fluid.set_state("position", converted_dtype)
        np.testing.assert_array_equal(fluid.positions(), converted_dtype.astype(np.float32))

        converted_layout = np.arange(18, dtype=np.float32).reshape(3, 6)[:, ::2]
        self.assertFalse(converted_layout.flags.c_contiguous)
        fluid.set_state("position", converted_layout)
        np.testing.assert_array_equal(fluid.positions(), converted_layout)

    def test_empty_arrays_are_valid_no_op_writes(self):
        fluid = Fluid(1, particle_count=1)
        fluid.set_state("temperature", np.array([7], dtype=np.float32))
        fluid.set_state("temperature", np.array([], dtype=np.float32), offset=1)
        np.testing.assert_array_equal(fluid.state("temperature"), [7])
