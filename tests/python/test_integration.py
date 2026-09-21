import gc
import unittest
from pathlib import Path

import numpy as np

from atlas import (
    Float3,
    Fluid,
    Sphere,
    System,
    UniformGenerator,
    Unit,
    Universe,
    VolumeSink,
    VolumeSource,
    get_default_engine,
)

from support import run_python


class IntegrationTests(unittest.TestCase):
    def test_source_motion_and_sink_keep_population_valid(self):
        source = VolumeSource(Unit(Sphere(Float3(0.15, 0.5, 0.5), 0.1)), spacing=0.1)
        generator = UniformGenerator([1], [0], bulk_velocity=Float3(1, 0, 0))
        sink = VolumeSink(Unit(Sphere(Float3(0.85, 0.5, 0.5), 0.15)))
        system = System(
            Fluid(128),
            Universe(Float3(0), Float3(1), 0.25),
            0.1,
            emitters=[(source, generator)],
            sinks=[sink],
        )
        del source, generator, sink
        gc.collect()
        counts = []
        for _ in range(8):
            system.update()
            counts.append(system.fluid.particle_count)
            self.assertLessEqual(system.fluid.particle_count, system.fluid.buffer_size)
            self.assertTrue(np.isfinite(system.fluid.positions()).all())
            self.assertTrue(np.isfinite(system.fluid.velocities()).all())
        self.assertTrue(any(count > 0 for count in counts))
        self.assertEqual(system.step, 8)

    def test_user_controls_numpy_output(self):
        system = System(
            Fluid.from_arrays(
                np.zeros((1, 3), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
            ),
            Universe(Float3(0), Float3(2), 1),
            0.1,
        )
        outputs = []
        for step in range(5):
            system.update()
            if step % 2 == 0:
                outputs.append(system.fluid.positions())
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(isinstance(value, np.ndarray) for value in outputs))
        np.testing.assert_allclose(outputs[-1], [[0.5, 0.5, 0.5]], atol=1e-6)

    def test_maintained_dsmc_example_smoke(self):
        example = Path(__file__).resolve().parents[2] / "examples/python/dsmc_dense_cell.py"
        result = run_python(f"""
            import runpy
            runpy.run_path({str(example)!r}, run_name="__main__")
        """, engine=get_default_engine(), timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("particles=200", result.stdout)
        self.assertIn("over 20 steps", result.stdout)
