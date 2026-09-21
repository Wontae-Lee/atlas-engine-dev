import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas import Float3, Fluid, System, Universe
from atlas.serialization import (
    load_fluid_binary,
    load_universe_binary,
    restore_fluid,
    restore_universe,
    save_fluid_binary,
    save_universe_binary,
)


class SerializationTests(unittest.TestCase):
    def test_fluid_round_trip_preserves_metadata_and_states(self):
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        velocities = -positions
        fluid = Fluid.from_arrays(positions, velocities, species=np.array([1, 0], dtype=np.uint64), buffer_size=4)
        fluid.set_state("temperature", np.array([10, 20], dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fluid.bin"
            save_fluid_binary(fluid, path)
            decoded = load_fluid_binary(path)
            restored = restore_fluid(path)
            loaded = Fluid.load(path)
            self.assertEqual(decoded["buffer_size"], 4)
            self.assertEqual(decoded["particle_count"], 2)
            self.assertEqual(decoded["positions"].shape, (4, 3))
            np.testing.assert_array_equal(decoded["positions"][:2], positions)
            for value in (restored, loaded):
                self.assertEqual(value.particle_count, 2)
                self.assertEqual(value.buffer_size, 4)
                np.testing.assert_array_equal(value.positions(), positions)
                np.testing.assert_array_equal(value.state("temperature"), [10, 20])

    def test_universe_round_trip_preserves_grid_and_states(self):
        universe = Universe(Float3(-1), Float3(1), 1)
        temperatures = np.arange(universe.cell_count, dtype=np.float32)
        velocities = np.arange(universe.cell_count * 3, dtype=np.float32).reshape(-1, 3)
        universe.set_state("temperature", temperatures)
        universe.set_state("bulk_velocity", velocities)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "universe.bin"
            save_universe_binary(universe, path)
            decoded = load_universe_binary(path)
            restored = restore_universe(path)
            loaded = Universe.load(path)
            self.assertEqual(decoded["lower_corner"], Float3(-1))
            self.assertEqual(decoded["upper_corner"], Float3(1))
            np.testing.assert_array_equal(decoded["temperature"], temperatures)
            for value in (restored, loaded):
                self.assertEqual(value.cell_count, universe.cell_count)
                np.testing.assert_array_equal(value.state("temperature"), temperatures)
                np.testing.assert_array_equal(value.state("bulk_velocity"), velocities)

    def test_saved_snapshots_are_independent_of_live_state(self):
        fluid = Fluid.from_arrays(
            np.array([[1, 2, 3]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fluid.bin"
            save_fluid_binary(fluid, path)
            fluid.set_state("position", np.array([[9, 9, 9]], dtype=np.float32))
            restored = restore_fluid(path)
            np.testing.assert_array_equal(restored.positions(), [[1, 2, 3]])

    def test_system_save_uses_step_directory_and_temporary_path(self):
        system = System(
            Fluid.from_arrays(np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)),
            Universe(Float3(0), Float3(1), 1),
            0.1,
        )
        system.update()
        with tempfile.TemporaryDirectory() as directory:
            system.save(directory)
            snapshot = Path(directory) / System.snapshot_directory_name(system.step)
            self.assertEqual(snapshot.name, "time_step_1")
            self.assertTrue((snapshot / "fluid.bin").is_file())
            self.assertTrue((snapshot / "universe.bin").is_file())
            np.testing.assert_array_equal(
                restore_fluid(snapshot / "fluid.bin").positions(),
                system.fluid.positions(),
            )
