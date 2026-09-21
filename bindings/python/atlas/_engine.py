import importlib
import importlib.util
import os
from threading import RLock


_ENGINES = ("tbb", "cuda")
_default_engine = None
_native = None
_lock = RLock()


def available_engines():
    return tuple(
        engine for engine in _ENGINES
        if importlib.util.find_spec(f"{__package__}._core_{engine}") is not None
    )


def _validate(engine):
    if engine not in _ENGINES:
        raise ValueError("engine must be 'tbb' or 'cuda'")
    if engine not in available_engines():
        raise RuntimeError(
            f"The '{engine}' engine is not installed. Install an Atlas wheel "
            f"containing the {engine.upper()} extension."
        )
    return engine


def get_default_engine():
    global _default_engine
    with _lock:
        if _default_engine is None:
            requested = os.environ.get("ATLAS_DEFAULT_ENGINE")
            installed = available_engines()
            if requested is None:
                if not installed:
                    raise RuntimeError("No Atlas engine is installed. Install an Atlas wheel first.")
                requested = installed[0]
            _default_engine = _validate(requested)
        return _default_engine


def set_default_engine(engine):
    global _default_engine
    with _lock:
        if engine not in _ENGINES:
            raise ValueError("engine must be 'tbb' or 'cuda'")
        if _native is not None and engine != _default_engine:
            raise RuntimeError(
                "The Atlas engine is already loaded. Set the default engine before "
                "importing any Atlas classes or submodules; use a new process to change engines."
            )
        _default_engine = _validate(engine)


def load_engine():
    global _native
    with _lock:
        if _native is None:
            engine = get_default_engine()
            try:
                native = importlib.import_module(f"{__package__}._core_{engine}")
            except ImportError as error:
                raise ImportError(f"Could not load the Atlas '{engine}' engine: {error}") from error
            if native.engine != engine:
                raise ImportError(f"Atlas engine mismatch: requested '{engine}', loaded '{native.engine}'")
            _native = native
        return _native
