import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    backends = {name.upper() for name in os.environ.get("ATLAS_WHEEL_BACKENDS", "TBB CUDA").split()}
    if not backends or not backends <= {"TBB", "CUDA"}:
        raise SystemExit("ATLAS_WHEEL_BACKENDS must contain TBB and/or CUDA")

    with TemporaryDirectory(prefix="atlas-wheels-") as temporary:
        stage = Path(temporary)

        def build(backend, *options):
            command = [
                sys.executable, "-m", "build", "--wheel", "--no-isolation",
                "--outdir", str(stage / backend.lower()),
                "-C", f"cmake.define.ATLAS_DEVICE_SYSTEM={backend}",
                "-C", f"build-dir=build/wheel-{backend.lower()}", *options,
            ]
            if backend == "CUDA" and "CMAKE_CUDA_ARCHITECTURES" in os.environ:
                command.extend(["-C", "cmake.define.CMAKE_CUDA_ARCHITECTURES="
                                + os.environ["CMAKE_CUDA_ARCHITECTURES"]])
            subprocess.run(command, check=True)
            wheel, = (stage / backend.lower()).glob("*.whl")
            return wheel

        def repair(backend, wheel):
            subprocess.run([
                "auditwheel", "repair", str(wheel), "-w", f"dist/{backend.lower()}",
                "--exclude", "libcuda.so.1",
            ], check=True)

        tbb_wheel = build("TBB", "-C", "cmake.define.ATLAS_PYTHON_TBB_EXTENSION=")
        if "TBB" in backends:
            repair("TBB", tbb_wheel)
        if "CUDA" in backends:
            with zipfile.ZipFile(tbb_wheel) as wheel:
                extension, = [
                    name for name in wheel.namelist()
                    if PurePosixPath(name).parent == PurePosixPath("atlas")
                    and PurePosixPath(name).name.startswith("_core_tbb.") and name.endswith(".so")
                ]
                native = stage / PurePosixPath(extension).name
                native.write_bytes(wheel.read(extension))
            cuda_wheel = build("CUDA", "-C", f"cmake.define.ATLAS_PYTHON_TBB_EXTENSION={native}")
            repair("CUDA", cuda_wheel)
    for backend in sorted(backends):
        for wheel in Path(f"dist/{backend.lower()}").glob("*.whl"):
            print(f"{wheel.stat().st_size / 1048576:.1f} MB  {wheel}")


if __name__ == "__main__":
    main()
