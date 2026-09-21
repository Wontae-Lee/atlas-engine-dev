import gc
import unittest

import numpy as np

from atlas import Float3, Fluid, Int3, SpatialHashingSearcher, System, Universe


class SearcherTests(unittest.TestCase):
    def test_explicit_grid_queries(self):
        searcher = SpatialHashingSearcher(Float3(0), 1, Int3(2, 2, 2))
        self.assertEqual(searcher.lower_corner, Float3(0))
        self.assertEqual(searcher.grid_size, Int3(2, 2, 2))
        self.assertEqual(searcher.cell_count, 8)
        self.assertEqual(searcher.cell_for(Float3(1.2, 0.2, 1.2)), Int3(1, 0, 1))
        self.assertEqual(searcher.linear_key(Int3(1, 0, 1)), 5)
        self.assertTrue(searcher.contains_cell(Int3(1, 1, 1)))
        self.assertFalse(searcher.contains_cell(Int3(2, 1, 1)))
        with self.assertRaises(ValueError):
            searcher.linear_key(Int3(2, 0, 0))

    def test_classification_arrays_and_cell_ranges(self):
        universe = Universe(Float3(0), Float3(1), 0.5)
        universe.set_state("number_particle", np.zeros(universe.cell_count, dtype=np.float32))
        fluid = Fluid.from_arrays(
            np.array([[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        )
        searcher = SpatialHashingSearcher(universe)
        searcher.classify(fluid, universe)
        self.assertEqual(searcher.particle_count, 3)
        np.testing.assert_array_equal(np.sort(searcher.cell_key()), [0, 0, 1])
        np.testing.assert_array_equal(np.sort(searcher.indices()), [0, 1, 2])
        self.assertEqual(universe.state("number_particle").sum(), 3)
        self.assertEqual(searcher.cell_start()[0], 0)
        self.assertEqual(searcher.cell_end()[0], 2)
        searcher.reset()
        self.assertEqual(searcher.particle_count, 0)

    def test_classify_rejects_mismatched_universe_grid(self):
        fluid = Fluid.from_arrays(
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )
        searcher = SpatialHashingSearcher(Universe(Float3(0), Float3(1), 0.5))
        with self.assertRaises(ValueError):
            searcher.classify(fluid, Universe(Float3(0), Float3(2), 0.5))

    def test_system_returns_its_owned_searcher_and_keeps_owner_alive(self):
        system = System(
            Fluid.from_arrays(np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)),
            Universe(Float3(0), Float3(1), 0.5),
            0.01,
        )
        searcher = system.searcher
        self.assertIs(searcher, system.searcher)
        system.search()
        self.assertEqual(searcher.particle_count, 1)
        searcher.reset()
        self.assertEqual(system.searcher.particle_count, 0)
        system.search()
        del system
        gc.collect()
        self.assertEqual(searcher.particle_count, 1)
        np.testing.assert_array_equal(searcher.indices(), [0])
