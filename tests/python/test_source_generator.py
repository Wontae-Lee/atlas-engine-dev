import unittest

import numpy as np

from atlas import (
    Float3,
    Fluid,
    Generator,
    GeneratorType,
    JitteringGenerator,
    MaxwellBoltzmannGenerator,
    MaxwellSigmaGenerator,
    Source,
    SourceType,
    Sphere,
    SurfaceSource,
    UniformGenerator,
    Unit,
    VolumeSource,
)


class SourceGeneratorTests(unittest.TestCase):
    def test_volume_and_surface_sources_spawn_without_changing_live_count(self):
        body = Unit(Sphere(Float3(0), 0.45))
        for cls, expected in ((VolumeSource, SourceType.volume), (SurfaceSource, SourceType.surface)):
            with self.subTest(source=cls.__name__):
                source = cls(body, spacing=0.25, tolerance=0.15)
                fluid = Fluid(256)
                self.assertIsInstance(source, Source)
                self.assertEqual(source.type, expected)
                self.assertGreater(source.cached_count, 0)
                self.assertEqual(source.unit.geometry.type, body.geometry.type)
                written = source.spawn(fluid, offset=2)
                self.assertGreater(written, 0)
                self.assertLessEqual(written, fluid.buffer_size - 2)
                self.assertEqual(fluid.particle_count, 0)
                self.assertTrue(np.isfinite(fluid.state("position", full=True)[2:2 + written]).all())
                source.advance(0.1)

    def test_generators_write_velocity_and_species_without_changing_live_count(self):
        cases = (
            (UniformGenerator, GeneratorType.uniform, {}),
            (JitteringGenerator, GeneratorType.jittering, {}),
            (MaxwellSigmaGenerator, GeneratorType.maxwell_sigma, {"sigma": 1}),
            (MaxwellBoltzmannGenerator, GeneratorType.maxwell_boltzmann, {"species_mass": [1]}),
        )
        for cls, expected_type, extra in cases:
            with self.subTest(generator=cls.__name__):
                generator = cls(
                    [1],
                    [0],
                    bulk_velocity=Float3(2, 3, 4),
                    seed=42,
                    **extra,
                )
                fluid = Fluid(8)
                written = generator.generate(fluid, offset=2, count=4)
                self.assertIsInstance(generator, Generator)
                self.assertEqual(generator.type, expected_type)
                self.assertEqual(generator.bulk_velocity, Float3(2, 3, 4))
                self.assertAlmostEqual(generator.temperature, 273.15, places=4)
                self.assertEqual(written, 4)
                self.assertEqual(fluid.particle_count, 0)
                np.testing.assert_array_equal(fluid.state("species", full=True)[2:6], [0, 0, 0, 0])
                self.assertTrue(np.isfinite(fluid.state("velocity", full=True)[2:6]).all())

    def test_uniform_generator_has_deterministic_exact_output(self):
        generator = UniformGenerator(
            [1], [0], min_value=0, max_value=0, bulk_velocity=Float3(2, -3, 4), seed=7)
        fluid = Fluid(3)
        self.assertEqual(generator.generate(fluid, 0, 3), 3)
        np.testing.assert_array_equal(fluid.state("velocity", full=True), [[2, -3, 4]] * 3)

    def test_seeded_random_generators_are_reproducible_within_backend(self):
        for cls, kwargs in (
            (JitteringGenerator, {"base_value": 1, "jitter_radius": 0.5}),
            (MaxwellSigmaGenerator, {"sigma": 1}),
            (MaxwellBoltzmannGenerator, {"species_mass": [1]}),
        ):
            first = cls([1], [0], seed=123, **kwargs)
            second = cls([1], [0], seed=123, **kwargs)
            left, right = Fluid(8), Fluid(8)
            first.generate(left, 0, 8)
            second.generate(right, 0, 8)
            with self.subTest(generator=cls.__name__):
                np.testing.assert_array_equal(
                    left.state("velocity", full=True), right.state("velocity", full=True))
