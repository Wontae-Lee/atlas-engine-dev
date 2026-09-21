import unittest

from atlas import (
    Atom,
    Ion,
    Material,
    MaterialDictionary,
    MaterialType,
    Molecule,
    Neutron,
    Solid,
)


class MaterialTests(unittest.TestCase):
    def test_material_leaf_values_and_types(self):
        cases = (
            (Molecule, MaterialType.molecule),
            (Atom, MaterialType.atom),
            (Ion, MaterialType.ion),
            (Neutron, MaterialType.neutron),
        )
        for cls, expected_type in cases:
            material = cls(2, 3, 4, 5, 6, 7, 8, 9)
            self.assertIsInstance(material, Material)
            self.assertEqual(material.type, expected_type)
            self.assertEqual(material.mass(), 2)
            self.assertEqual(material.translational_energy(), 3)
            self.assertEqual(material.rotational_energy(), 4)
            self.assertEqual(material.vibrational_energy(), 5)
        self.assertEqual(Solid(10).type, MaterialType.solid)
        self.assertEqual(Solid(10).mass(), 10)

    def test_dictionary_indexing_replacement_and_snapshots(self):
        dictionary = MaterialDictionary([Solid(1), Solid(2)])
        self.assertEqual(len(dictionary), 2)
        self.assertFalse(dictionary.empty())
        self.assertEqual(dictionary[-1].mass(), 2)
        snapshot = dictionary.materials()
        dictionary[-1] = Solid(3)
        self.assertEqual(dictionary[1].mass(), 3)
        self.assertEqual(snapshot[1].mass(), 2)
        dictionary.set_materials([Solid(4)])
        self.assertEqual(len(dictionary), 1)
        self.assertEqual(dictionary[0].mass(), 4)
        with self.assertRaises(IndexError):
            _ = dictionary[1]
        with self.assertRaises(IndexError):
            dictionary[1] = Solid(5)
