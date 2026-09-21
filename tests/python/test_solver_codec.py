import unittest

import numpy as np

from atlas import (
    Codec,
    CodecType,
    DsmcKernelType,
    DsmcSolver,
    Float3,
    KnudsenCodec,
    MaterialDictionary,
    Molecule,
    Solver,
    SolverType,
    System,
    Fluid,
    Universe,
)


class SolverCodecTests(unittest.TestCase):
    def test_dsmc_solver_configuration(self):
        for kernel in (
            DsmcKernelType.hard_sphere,
            DsmcKernelType.variable_hard_sphere,
            DsmcKernelType.variable_soft_sphere,
        ):
            solver = DsmcSolver(
                kernel_type=kernel,
                majorant_sample_pairs=3,
                majorant_exhaustive_limit=4,
            )
            self.assertIsInstance(solver, Solver)
            self.assertEqual(solver.type, SolverType.dsmc)
            self.assertEqual(solver.kernel_type, kernel)
            self.assertEqual(solver.majorant_sample_pairs, 3)
            self.assertEqual(solver.majorant_exhaustive_limit, 4)

    def test_invalid_solver_parameters_raise(self):
        with self.assertRaises(RuntimeError):
            DsmcSolver(majorant_sample_pairs=0)
        with self.assertRaises(RuntimeError):
            DsmcSolver(majorant_exhaustive_limit=1)

    def test_dsmc_solver_participates_in_system_update(self):
        positions = np.array(
            [[0.2, 0.2, 0.2], [0.3, 0.2, 0.2], [0.4, 0.2, 0.2]],
            dtype=np.float32,
        )
        velocities = np.array(
            [[300, 0, 0], [-200, 100, 0], [0, -100, 250]],
            dtype=np.float32,
        )
        materials = MaterialDictionary([
            Molecule(4.65e-26, 0, 0, 0, 4.17e-10, 273, 0.74, 1),
        ])
        system = System(
            Fluid.from_arrays(
                positions,
                velocities,
                statistical_weight=1e22,
                materials=materials,
            ),
            Universe(Float3(0), Float3(0.9), 1),
            1e-4,
            solvers=[DsmcSolver(kernel_type=DsmcKernelType.hard_sphere)],
        )
        system.update()
        self.assertGreater(system.universe.state("collision_count").sum(), 0)
        self.assertTrue(np.isfinite(system.fluid.velocities()).all())
        self.assertFalse(np.array_equal(system.fluid.velocities(), velocities))

    def test_knudsen_codec_allocates_solver_indices(self):
        codec = KnudsenCodec()
        self.assertIsInstance(codec, Codec)
        self.assertEqual(codec.type, CodecType.knudsen)
        self.assertEqual(codec.split_count, 4)
        self.assertEqual(codec.knudsen_number(0), 0)
        self.assertEqual(codec.solver_index(-1), 0)

        universe = Universe(Float3(0), Float3(1), 1)
        counts = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000], dtype=np.float32)
        universe.set_state("number_particle", counts)
        universe.set_state("allocated_solver", np.full(universe.cell_count, -1, dtype=np.int32))
        codec.allocate(universe)
        expected = np.array(
            [codec.solver_index(codec.knudsen_number(float(value))) for value in counts],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(universe.state("allocated_solver"), expected)

    def test_invalid_codec_parameters_raise(self):
        keywords = (
            "representative_characteristic_length",
            "representative_collision_cross_sectional_area",
            "representative_statistical_weight",
            "representative_cell_volume",
        )
        for keyword in keywords:
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    KnudsenCodec(**{keyword: 0})
