import importlib
import inspect
import unittest
from importlib.metadata import version

import atlas
from atlas import Float3, Fluid, System, Universe
from atlas import _core


PUBLIC_MODULES = (
    "math",
    "random",
    "spatial",
    "sync",
    "geometry",
    "unit",
    "material",
    "fluid",
    "universe",
    "source",
    "generator",
    "solver",
    "codec",
    "collider",
    "sink",
    "system",
    "searcher",
    "sampling",
    "serialization",
)


class ImportTests(unittest.TestCase):
    def test_root_imports_and_distribution_version(self):
        self.assertTrue(all(inspect.isclass(value) for value in (Float3, Fluid, Universe, System)))
        self.assertEqual(atlas.__version__, version("atlas-engine"))

    def test_public_submodules_reexport_native_objects(self):
        for name in PUBLIC_MODULES:
            with self.subTest(module=name):
                public = importlib.import_module("atlas." + name)
                native = getattr(_core, name)
                self.assertTrue(public.__all__)
                for symbol in public.__all__:
                    self.assertIs(getattr(public, symbol), getattr(native, symbol))

    def test_registered_classes_use_pascal_case(self):
        for name in PUBLIC_MODULES:
            public = importlib.import_module("atlas." + name)
            for symbol in public.__all__:
                value = getattr(public, symbol)
                if inspect.isclass(value):
                    with self.subTest(module=name, symbol=symbol):
                        self.assertTrue(symbol[0].isupper())
                        self.assertNotIn("_", symbol)
                        self.assertEqual(value.__name__, symbol)

    def test_nested_imports_preserve_type_identity(self):
        cases = (
            ("math.vector", "math", ("Bool3", "Int3", "Float3")),
            ("math.matrix", "math", ("Float3x3",)),
            ("spatial.bounding_volume_hierarchy", "spatial", ("BVH", "LBVH", "SAHBVH")),
            ("solver.dsmc", "solver", ("DsmcSolver",)),
            ("solver.dsmc.kernel", "solver", ("DsmcKernel", "DsmcKernelType")),
        )
        for path, parent, symbols in cases:
            with self.subTest(path=path):
                nested = importlib.import_module("atlas." + path)
                module = importlib.import_module("atlas." + parent)
                for symbol in symbols:
                    self.assertIs(getattr(nested, symbol), getattr(module, symbol))

    def test_binding_helpers_are_not_public_modules(self):
        self.assertNotIn("detail", atlas.__all__)
        self.assertFalse(hasattr(atlas, "detail"))
        self.assertFalse(hasattr(atlas, "DeviceBuffer"))
        self.assertFalse(hasattr(atlas, "HostBuffer"))
