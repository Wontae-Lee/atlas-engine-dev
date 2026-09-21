import unittest

from atlas import (
    Collider,
    ColliderType,
    Float3,
    IsothermalCollider,
    Sink,
    SinkType,
    Sphere,
    SurfaceSink,
    TracingSink,
    Unit,
    VolumeSink,
)


class BoundaryTests(unittest.TestCase):
    def test_isothermal_collider_exposes_leaf_specific_reflection(self):
        collider = IsothermalCollider(
            Unit(Sphere(Float3(0), 1)),
            momentum_accommodation_coefficient=0,
            restitution=1,
        )
        self.assertIsInstance(collider, Collider)
        self.assertEqual(collider.type, ColliderType.isothermal)
        self.assertFalse(hasattr(Collider, "reflect"))
        self.assertTrue(hasattr(collider, "reflect"))
        self.assertEqual(collider.momentum_accommodation_coefficient, 0)
        self.assertEqual(
            collider.reflect(Float3(0, 0, -1), Float3(0, 0, 1)),
            Float3(0, 0, 1),
        )

    def test_collider_trace_and_collide_are_deterministic_for_specular_wall(self):
        collider = IsothermalCollider(
            Unit(Sphere(Float3(0), 1)),
            momentum_accommodation_coefficient=0,
        )
        hit = collider.trace(Float3(0, 0, 2), Float3(0, 0, -2), 1)
        self.assertTrue(hit.is_intersecting)
        position, velocity = collider.collide(
            hit, Float3(0, 0, 2), Float3(0, 0, -2), 1)
        self.assertTrue(position.z > 1)
        self.assertEqual(velocity, Float3(0, 0, 2))
        self.assertTrue(collider.bound().contains(Float3(0)))

    def test_sink_predicates(self):
        body = Unit(Sphere(Float3(0), 1))
        cases = (
            (VolumeSink(body), SinkType.volume, Float3(0), Float3(0), True),
            (VolumeSink(body), SinkType.volume, Float3(2, 0, 0), Float3(0), False),
            (SurfaceSink(body, tolerance=0.01), SinkType.surface, Float3(1, 0, 0), Float3(0), True),
            (TracingSink(body), SinkType.tracing, Float3(2, 0, 0), Float3(-2, 0, 0), True),
        )
        for sink, expected_type, position, velocity, expected in cases:
            with self.subTest(type=expected_type, position=position):
                self.assertIsInstance(sink, Sink)
                self.assertEqual(sink.type, expected_type)
                self.assertEqual(sink.despawn(position, velocity, 1), expected)
                sink.advance(0.1)

    def test_invalid_boundary_parameters_raise(self):
        body = Unit(Sphere(Float3(0), 1))
        with self.assertRaises(RuntimeError):
            IsothermalCollider(body, momentum_accommodation_coefficient=-0.1)
        with self.assertRaises(RuntimeError):
            IsothermalCollider(body, momentum_accommodation_coefficient=1.1)
        with self.assertRaises(RuntimeError):
            IsothermalCollider(body, restitution=-1)
        with self.assertRaises(RuntimeError):
            VolumeSink(body, tolerance=-1)
        with self.assertRaises(RuntimeError):
            SurfaceSink(body, tolerance=-1)
