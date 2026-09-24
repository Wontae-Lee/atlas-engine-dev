import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    dist = root / "dist/python-validation"
    source = dist / "source"
    wheels = dist / "wheels"
    for directory, pattern in ((source, "*.tar.gz"), (wheels, "*.whl")):
        directory.mkdir(parents=True, exist_ok=True)
        for artifact in directory.glob(pattern):
            artifact.unlink()

    def python(*args, **kwargs):
        subprocess.run([sys.executable, *map(str, args)], check=True, **kwargs)

    python("-m", "build", "--sdist", "--no-isolation", "--outdir", source)
    sdist, = source.glob("*.tar.gz")
    python("-m", "pip", "wheel", "--no-build-isolation", "--no-deps", sdist,
           "--wheel-dir", wheels, "-C", "cmake.define.ATLAS_DEVICE_SYSTEM=TBB",
           "-C", "cmake.define.ATLAS_HOST_COMPILER=native")
    wheel, = wheels.glob("*.whl")
    python("-m", "twine", "check", "--strict", sdist, wheel)
    python("-m", "pip", "install", "--force-reinstall", "--no-deps", wheel)
    python("-m", "pip", "check")
    env = dict(os.environ, ATLAS_DEFAULT_ENGINE="tbb", PYTHONDONTWRITEBYTECODE="1")
    python("-c", "from importlib.metadata import version; import atlas; from atlas import _core; "
           "assert atlas.__version__ == version('atlas-engine'); "
           "assert atlas.available_engines() == ('tbb',); "
           "assert atlas.get_default_engine() == _core.engine == 'tbb'; "
           "print(f'Atlas {atlas.__version__}: {_core.engine}')", env=env)
    python("-X", "faulthandler", "-m", "unittest", "discover", "-s", "tests/python", "-v",
           env=env, timeout=300)
    python("-X", "faulthandler", "examples/python/main.py", env=env, timeout=60)


if __name__ == "__main__":
    main()
