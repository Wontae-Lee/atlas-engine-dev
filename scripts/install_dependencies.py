import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    build = Path(os.environ.get("ATLAS_DEPS_BUILD_DIR", root / "build/dependencies"))
    prefix = os.environ.get("ATLAS_DEPS_PREFIX", "/opt/atlas-deps")
    os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    subprocess.run([
        "cmake", "-S", str(root / "cmake/dependencies"), "-B", str(build), "-G", "Ninja",
        f"-DCMAKE_INSTALL_PREFIX={prefix}", *sys.argv[1:],
    ], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "1"], check=True)


if __name__ == "__main__":
    main()
