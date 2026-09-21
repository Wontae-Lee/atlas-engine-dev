import unittest

import numpy as np

from atlas import (
    DsmcSolver,
    Float3,
    Fluid,
    KnudsenCodec,
    MaterialDictionary,
    Solid,
    Sphere,
    System,
    UniformGenerator,
    Unit,
    Universe,
    VolumeSource,
)


class ErrorTests(unittest.TestCase):
    def test_unknown_fluid_state_operations_raise_key_error(self):
        fluid = Fluid(1)
        operations = (
            lambda: fluid.state("does_not_exist"),
            lambda: fluid.set_state("does_not_exist", np.zeros(1, dtype=np.float32)),
            lambda: fluid.has_state("does_not_exist"),
            lambda: fluid.remove_state("does_not_exist"),
            lambda: fluid.reset_state("does_not_exist"),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(KeyError):
                    operation()

    def test_unknown_universe_state_operations_raise_key_error(self):
        universe = Universe(Float3(0), Float3(1), 1)
        operations = (
            lambda: universe.state("does_not_exist"),
            lambda: universe.set_state("does_not_exist", np.zeros(1, dtype=np.float32)),
            lambda: universe.has_state("does_not_exist"),
            lambda: universe.remove_state("does_not_exist"),
            lambda: universe.reset_state("does_not_exist"),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(KeyError):
                    operation()

    def test_reset_absent_state_raises_key_error(self):
        with self.assertRaises(KeyError):
            Fluid(1).reset_state("temperature")
        with self.assertRaises(KeyError):
            Universe(Float3(0), Float3(1), 1).reset_state("temperature")

    def test_invalid_fluid_parameters_raise(self):
        with self.assertRaises(RuntimeError):
            Fluid(buffer_size=1, particle_count=2)
        with self.assertRaises(RuntimeError):
            Fluid(buffer_size=1, statistical_weight=0)
        fluid = Fluid(1)
        with self.assertRaises(IndexError):
            fluid.set_particle_count(2)
        with self.assertRaises(ValueError):
            fluid.set_active(np.array([2], dtype=np.int32))

    def test_invalid_species_id_is_rejected(self):
        fluid = Fluid(2, particle_count=1, materials=MaterialDictionary([Solid(1)]))
        with self.assertRaises(ValueError):
            fluid.set_state("species", np.array([1], dtype=np.uint64))
        with self.assertRaises((TypeError, ValueError)):
            fluid.set_state("species", np.array([-1], dtype=np.int64))

    def test_invalid_generator_source_solver_codec_and_system_raise(self):
        with self.assertRaises(RuntimeError):
            UniformGenerator([], [])
        with self.assertRaises(RuntimeError):
            UniformGenerator([1], [0, 1])
        with self.assertRaises(RuntimeError):
            VolumeSource(Unit(Sphere(Float3(0), 1)), spacing=0)
        with self.assertRaises(RuntimeError):
            DsmcSolver(majorant_sample_pairs=0)
        with self.assertRaises(ValueError):
            KnudsenCodec(representative_cell_volume=0)
        with self.assertRaises(ValueError):
            System(
                Fluid(2),
                Universe(Float3(0), Float3(1), 1),
                0.1,
                source=VolumeSource(Unit(Sphere(Float3(0), 1))),
            )
