import tempfile
import unittest
from pathlib import Path

import numpy as np

from atlas import (
    BVH,
    Box,
    Circle,
    Cylinder,
    Float3,
    Geometry,
    GeometryType,
    LBVH,
    Plane,
    PolygonalPrism,
    Ray,
    SAHBVH,
    Sphere,
    Square,
    Triangle,
    TriangleMesh,
    Unit,
)


class GeometryTests(unittest.TestCase):
    def test_exposed_geometry_types_and_bounds(self):
        cases = (
            (Sphere(Float3(0), 1), GeometryType.sphere),
            (Plane(Float3(0, 0, 1), 0), GeometryType.plane),
            (Box(Float3(-1), Float3(1)), GeometryType.box),
            (Cylinder(Float3(0), 1, 2), GeometryType.cylinder),
            (Circle(Float3(0), Float3(0, 0, 1), 1), GeometryType.circle),
            (Square(Float3(0), Float3(0, 0, 1), 1), GeometryType.square),
            (Triangle(Float3(0), Float3(1, 0, 0), Float3(0, 1, 0)), GeometryType.triangle),
            (PolygonalPrism(Float3(0), 6, 1, 2), GeometryType.polygonal_prism),
        )
        for geometry, expected_type in cases:
            with self.subTest(type=expected_type):
                self.assertIsInstance(geometry, Geometry)
                self.assertEqual(geometry.type, expected_type)
                self.assertTrue(geometry.is_valid())
                self.assertTrue(geometry.bound().is_valid())

    def test_sphere_queries_are_analytic(self):
        sphere = Sphere(Float3(0), 1)
        self.assertTrue(sphere.is_inside(Float3(0)))
        self.assertFalse(sphere.is_inside(Float3(2, 0, 0)))
        self.assertAlmostEqual(sphere.signed_distance(Float3(2, 0, 0)), 1)
        hit = sphere.trace(Ray(Float3(-2, 0, 0), Float3(1, 0, 0)))
        self.assertTrue(hit.is_intersecting)
        self.assertAlmostEqual(hit.distance, 1)
        self.assertEqual(hit.point, Float3(-1, 0, 0))

    def test_unit_applies_pose_to_geometry(self):
        from atlas import Sync

        unit = Unit(Sphere(Float3(0), 1), Sync(translation=Float3(3, 0, 0)))
        self.assertTrue(unit.world_bound().contains(Float3(3, 0, 0)))
        self.assertFalse(unit.world_bound().contains(Float3(0)))
        hit = unit.trace(Ray(Float3(0), Float3(1, 0, 0)))
        self.assertTrue(hit.is_intersecting)
        self.assertAlmostEqual(hit.distance, 2)

    def test_triangle_mesh_constructor_overloads(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)
        triangle = Triangle(*(Float3(*row) for row in vertices))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triangle.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            meshes = (TriangleMesh([triangle]), TriangleMesh(vertices, indices), TriangleMesh(str(path)))
            for mesh in meshes:
                with self.subTest(mesh=mesh):
                    self.assertIs(type(mesh), TriangleMesh)
                    self.assertEqual(mesh.type, GeometryType.triangle_mesh)
                    hit = mesh.trace(Ray(Float3(0.25, 0.25, 1), Float3(0, 0, -1)))
                    self.assertTrue(hit.is_intersecting)
                    self.assertAlmostEqual(hit.distance, 1)
        with self.assertRaises(IndexError):
            TriangleMesh(vertices, np.array([[0, 1, 3]], dtype=np.int64))

    def test_bvh_builders_cover_triangle(self):
        triangle = Triangle(Float3(0), Float3(1, 0, 0), Float3(0, 1, 0))
        for cls in (LBVH, SAHBVH):
            hierarchy = cls([triangle])
            self.assertIsInstance(hierarchy, BVH)
            self.assertEqual(hierarchy.indices(), [0])
            self.assertTrue(hierarchy.bounds()[0].contains(Float3(0.25, 0.25, 0)))
