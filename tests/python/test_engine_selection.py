import importlib.util
import unittest

import atlas

from support import run_python


class EngineSelectionTests(unittest.TestCase):
    def assert_python_ok(self, source, engine=None):
        result = run_python(source, engine)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_and_queries_are_lazy(self):
        self.assert_python_ok("""
            import sys
            import atlas

            engines = atlas.available_engines()
            assert engines
            assert isinstance(engines, tuple)
            assert set(engines) <= {"tbb", "cuda"}
            assert atlas.get_default_engine() in engines
            assert "atlas._core_tbb" not in sys.modules
            assert "atlas._core_cuda" not in sys.modules
        """)

    def test_available_engines_match_installed_extensions(self):
        installed = {
            name for name in ("tbb", "cuda")
            if importlib.util.find_spec("atlas._core_" + name) is not None
        }
        self.assertEqual(set(atlas.available_engines()), installed)

    def test_each_installed_engine_can_be_selected_by_api_and_environment(self):
        for engine in atlas.available_engines():
            with self.subTest(engine=engine):
                self.assert_python_ok(f"""
                    import atlas
                    atlas.set_default_engine({engine!r})
                    assert atlas.get_default_engine() == {engine!r}
                    from atlas import Float3, _core
                    assert _core.engine == {engine!r}
                    assert Float3(1, 2, 3).z == 3
                """)
                self.assert_python_ok("""
                    import os
                    import atlas
                    expected = os.environ["ATLAS_DEFAULT_ENGINE"]
                    assert atlas.get_default_engine() == expected
                    from atlas import Float3, _core
                    assert _core.engine == expected
                    assert Float3(1).x == 1
                """, engine)

    def test_native_import_pins_the_engine(self):
        for engine in atlas.available_engines():
            with self.subTest(engine=engine):
                self.assert_python_ok(f"""
                    import atlas
                    atlas.set_default_engine({engine!r})
                    from atlas import Float3
                    atlas.set_default_engine({engine!r})
                    other = "cuda" if {engine!r} == "tbb" else "tbb"
                    try:
                        atlas.set_default_engine(other)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("loaded engine changed")
                    assert Float3(2).x == 2
                """)

    def test_invalid_engine_names_and_environment_are_rejected(self):
        self.assert_python_ok("""
            import atlas
            expected = atlas.get_default_engine()
            for name in ("cpu", "", "opencl"):
                try:
                    atlas.set_default_engine(name)
                except ValueError:
                    pass
                else:
                    raise AssertionError(name)
                assert atlas.get_default_engine() == expected
        """)
        self.assert_python_ok("""
            try:
                import atlas
                atlas.get_default_engine()
            except ValueError:
                pass
            else:
                raise AssertionError("invalid environment accepted")
        """, "cpu")

    def test_missing_engine_is_not_silently_replaced(self):
        missing = {"tbb", "cuda"} - set(atlas.available_engines())
        if not missing:
            self.skipTest("Both native engines are installed")
        for engine in sorted(missing):
            with self.subTest(engine=engine):
                self.assert_python_ok(f"""
                    import atlas
                    try:
                        atlas.set_default_engine({engine!r})
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("missing engine selected")
                """)
