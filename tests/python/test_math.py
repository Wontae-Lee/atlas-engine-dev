import unittest

from atlas.math import Bool3, Float3, Float3x3, Int3, Quaternion, cross, dot


class MathTests(unittest.TestCase):
    def test_bool_components_and_explicit_reduction(self):
        value = Bool3(True, False, True)
        self.assertTrue(value.any())
        self.assertFalse(value.all())
        self.assertTrue((value & ~value).none())
        value.y = True
        self.assertTrue(value.all())
        with self.assertRaises(TypeError):
            bool(value)

    def test_vector_arithmetic_and_checked_indexing(self):
        value = Float3(1, 2, 3)
        self.assertEqual(dot(value, value), 14)
        self.assertEqual(cross(Float3(1, 0, 0), Float3(0, 1, 0)), Float3(0, 0, 1))
        self.assertEqual(value * 2, Float3(2, 4, 6))
        for vector in (value, Int3(1, 2, 3)):
            with self.subTest(type=type(vector).__name__):
                self.assertEqual(vector[-1], 3)
                vector[0] = 4
                self.assertEqual(vector.x, 4)
                with self.assertRaises(IndexError):
                    _ = vector[3]
                with self.assertRaises(IndexError):
                    vector[-4] = 0

    def test_matrix_and_quaternion_identity(self):
        value = Float3(1, 2, 3)
        self.assertEqual(Float3x3(1) * value, value)
        self.assertEqual(Quaternion().rotate(value), value)
        self.assertIsNone(Float3x3(0).try_inverse())
