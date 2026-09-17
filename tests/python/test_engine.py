import os
import subprocess
import sys
import textwrap
import unittest

import atlas


class EngineTests(unittest.TestCase):
    def run_python(self, source, engine=None):
        environment = os.environ.copy()
        environment.pop("ATLAS_DEFAULT_ENGINE", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if engine is not None:
            environment["ATLAS_DEFAULT_ENGINE"] = engine
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(source)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_and_engine_queries_do_not_load_native_extension(self):
        self.run_python("""
            import sys
            import atlas

            engines = atlas.available_engines()
            assert engines
            assert isinstance(engines, tuple)
            assert set(engines) <= {"tbb", "cuda"}
            assert len(engines) == len(set(engines))
            assert atlas.get_default_engine() in engines
            assert "atlas._core_tbb" not in sys.modules
            assert "atlas._core_cuda" not in sys.modules
            assert "atlas.math" not in sys.modules
        """)

    def test_available_engines_match_installed_extensions(self):
        self.run_python("""
            import importlib.util
            import atlas

            installed = {
                engine for engine in ("tbb", "cuda")
                if importlib.util.find_spec("atlas._core_" + engine) is not None
            }
            assert set(atlas.available_engines()) == installed
        """)

    def test_default_prefers_tbb_when_available(self):
        self.run_python("""
            import atlas

            expected = "tbb" if "tbb" in atlas.available_engines() else "cuda"
            assert atlas.get_default_engine() == expected
            from atlas import Float3, _core
            assert _core.engine == expected
            assert Float3(1, 2, 3).z == 3
        """)

    def test_environment_selects_each_installed_engine(self):
        for engine in atlas.available_engines():
            with self.subTest(engine=engine):
                self.run_python("""
                    import os
                    import atlas

                    expected = os.environ["ATLAS_DEFAULT_ENGINE"]
                    assert atlas.get_default_engine() == expected
                    from atlas import Float3, _core
                    assert _core.engine == expected
                    assert Float3(1, 2, 3).x == 1
                """, engine=engine)

    def test_api_selection_preserves_root_and_nested_class_identity(self):
        for engine in atlas.available_engines():
            with self.subTest(engine=engine):
                self.run_python(f"""
                    import inspect
                    import sys
                    import atlas

                    atlas.set_default_engine({engine!r})
                    assert atlas.get_default_engine() == {engine!r}
                    assert "atlas._core_tbb" not in sys.modules
                    assert "atlas._core_cuda" not in sys.modules
                    from atlas import Float3, Sphere, _core
                    from atlas.math import Float3 as ModuleFloat3
                    from atlas.math.vector import Float3 as VectorFloat3
                    from atlas.geometry import Sphere as ModuleSphere

                    assert _core.engine == {engine!r}
                    assert Float3 is ModuleFloat3 is VectorFloat3 is _core.math.Float3
                    assert Sphere is ModuleSphere is _core.geometry.Sphere
                    assert inspect.isclass(Sphere)
                    assert type(Float3(1, 2, 3)) is Float3
                    other = "cuda" if {engine!r} == "tbb" else "tbb"
                    assert "atlas._core_" + other not in sys.modules
                """)

    def test_submodule_import_pins_engine_before_root_class_access(self):
        for engine in atlas.available_engines():
            with self.subTest(engine=engine):
                self.run_python(f"""
                    import unittest
                    import atlas

                    atlas.set_default_engine({engine!r})
                    from atlas.math.vector import Float3
                    atlas.set_default_engine({engine!r})
                    other = "cuda" if {engine!r} == "tbb" else "tbb"
                    with unittest.TestCase().assertRaises(RuntimeError):
                        atlas.set_default_engine(other)
                    assert atlas.get_default_engine() == {engine!r}
                    assert atlas.Float3 is Float3
                    assert Float3(1, 2, 3).y == 2
                """)

    def test_root_class_import_pins_engine(self):
        for engine in atlas.available_engines():
            with self.subTest(engine=engine):
                self.run_python(f"""
                    import unittest
                    import atlas

                    atlas.set_default_engine({engine!r})
                    from atlas import Float3
                    atlas.set_default_engine({engine!r})
                    other = "cuda" if {engine!r} == "tbb" else "tbb"
                    with unittest.TestCase().assertRaises(RuntimeError):
                        atlas.set_default_engine(other)
                    assert atlas.get_default_engine() == {engine!r}
                    assert atlas.Float3 is Float3
                """)

    def test_engine_can_change_until_native_import(self):
        if len(atlas.available_engines()) < 2:
            self.skipTest("Both engine extensions are required")
        self.run_python("""
            import atlas

            assert atlas.get_default_engine() == "cuda"
            atlas.set_default_engine("cuda")
            assert atlas.get_default_engine() == "cuda"
            atlas.set_default_engine("tbb")
            assert atlas.get_default_engine() == "tbb"
            from atlas import Float3, _core
            assert _core.engine == "tbb"
            assert Float3(2).z == 2
        """, engine="cuda")

    def test_invalid_engine_is_rejected_without_changing_default(self):
        self.run_python("""
            import unittest
            import atlas

            expected = atlas.get_default_engine()
            for engine in ("cpu", "", "opencl"):
                with unittest.TestCase().assertRaises(ValueError):
                    atlas.set_default_engine(engine)
                assert atlas.get_default_engine() == expected
            from atlas import Float3, _core
            assert _core.engine == expected
            assert Float3(2).x == 2
        """)

    def test_invalid_environment_engine_is_rejected(self):
        self.run_python("""
            import unittest

            with unittest.TestCase().assertRaises(ValueError):
                import atlas
                atlas.get_default_engine()
        """, engine="cpu")

    def test_missing_engine_is_rejected_without_fallback(self):
        missing = set(("tbb", "cuda")) - set(atlas.available_engines())
        if not missing:
            self.skipTest("Both engine extensions are installed")
        for engine in sorted(missing):
            with self.subTest(engine=engine):
                self.run_python(f"""
                    import unittest
                    import atlas

                    expected = atlas.get_default_engine()
                    with unittest.TestCase().assertRaises(RuntimeError):
                        atlas.set_default_engine({engine!r})
                    assert atlas.get_default_engine() == expected
                    from atlas import Float3, _core
                    assert _core.engine == expected
                    assert Float3(2).x == 2
                """)

    def test_missing_environment_engine_is_rejected_without_fallback(self):
        missing = set(("tbb", "cuda")) - set(atlas.available_engines())
        if not missing:
            self.skipTest("Both engine extensions are installed")
        for engine in sorted(missing):
            with self.subTest(engine=engine):
                self.run_python("""
                    import unittest

                    with unittest.TestCase().assertRaises(RuntimeError):
                        import atlas
                        atlas.get_default_engine()
                """, engine=engine)


if __name__ == "__main__":
    unittest.main()
