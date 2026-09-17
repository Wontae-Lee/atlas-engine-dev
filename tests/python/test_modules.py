import importlib
import inspect
import unittest
from importlib.metadata import version

import atlas
from atlas import _core


MODULES = (
    "math", "random", "spatial", "sync", "geometry", "unit", "material",
    "fluid", "universe", "source", "generator", "solver", "codec",
    "collider", "sink", "observer", "system", "searcher", "sampling",
    "serialization",
)


class ModuleTests(unittest.TestCase):
    def test_installed_distribution_version(self):
        self.assertEqual(atlas.__version__, version("atlas-engine"))

    def test_public_exports_are_native_objects(self):
        for name in MODULES:
            with self.subTest(module=name):
                public = importlib.import_module("atlas." + name)
                native = getattr(_core, name)
                self.assertTrue(public.__all__)
                for symbol in public.__all__:
                    self.assertIs(getattr(public, symbol), getattr(native, symbol))

    def test_registered_class_names_are_pascal_case(self):
        for name in MODULES:
            public = importlib.import_module("atlas." + name)
            for symbol in public.__all__:
                value = getattr(public, symbol)
                if inspect.isclass(value):
                    with self.subTest(module=name, symbol=symbol):
                        self.assertTrue(symbol[0].isupper())
                        self.assertNotIn("_", symbol)
                        self.assertEqual(value.__name__, symbol)

    def test_nested_imports_preserve_type_identity(self):
        for path, parent, symbols in (
            ("math.vector", "math", ("Bool3", "Int3", "Float3")),
            ("math.matrix", "math", ("Float3x3",)),
            ("spatial.bounding_volume_hierarchy", "spatial", ("BVH", "LBVH", "SAHBVH")),
            ("solver.dsmc", "solver", ("DsmcSolver",)),
            ("solver.dsmc.kernel", "solver", ("DsmcKernel", "DsmcKernelType")),
        ):
            with self.subTest(path=path):
                nested = importlib.import_module("atlas." + path)
                module = importlib.import_module("atlas." + parent)
                for symbol in symbols:
                    self.assertIs(getattr(nested, symbol), getattr(module, symbol))
