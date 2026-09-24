import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Build or enter the Atlas Docker development environment.")
    parser.add_argument("action", choices=["build", "tbb", "cuda"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.arguments
    backend = args.action
    if args.action == "build":
        if not command or command[0] not in ("tbb", "cuda"):
            parser.error("build requires a tbb or cuda backend")
        backend, *command = command

    root = Path(__file__).resolve().parents[1]
    ubuntu = os.environ.get("UBUNTU_VERSION", "22.04")
    image = os.environ.get("ATLAS_DEV_IMAGE", f"atlas:{backend}-dev-ubuntu{ubuntu}")
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    if args.action == "build":
        docker = [
            "docker", "build", "--target", f"{backend}-dev", "-t", image,
            "--build-arg", f"UBUNTU_VERSION={ubuntu}", "--build-arg", f"BUILD_JOBS={jobs}",
            *command, str(root),
        ]
    else:
        cache = root / "build" / f"docker-{backend}-ubuntu{ubuntu}"
        for directory in ("build", "ccache", "pip"):
            (cache / directory).mkdir(parents=True, exist_ok=True)
        docker = [
            "docker", "run", "--rm", "--init", "--user", f"{os.getuid()}:{os.getgid()}",
            "-w", "/workspace",
            "--mount", f"type=bind,source={root},target=/workspace",
            "--mount", f"type=bind,source={cache / 'build'},target=/workspace/build",
            "--mount", f"type=bind,source={cache / 'ccache'},target=/tmp/atlas-ccache",
            "--mount", f"type=bind,source={cache / 'pip'},target=/tmp/atlas-pip-cache",
            "-e", "PIP_CACHE_DIR=/tmp/atlas-pip-cache",
            "-e", f"CMAKE_BUILD_PARALLEL_LEVEL={jobs}", "-e", "PYTHONDONTWRITEBYTECODE=1",
        ]
        if backend == "cuda" and os.environ.get("ATLAS_DOCKER_GPU", "1") == "1":
            docker.extend(["--gpus", "all"])
        if os.environ.get("ATLAS_DOCKER_DISPLAY") == "1":
            if not os.environ.get("DISPLAY") or not Path("/tmp/.X11-unix").is_dir():
                parser.error("ATLAS_DOCKER_DISPLAY=1 requires DISPLAY and the host X11 socket")
            docker.extend([
                "-e", "DISPLAY", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display",
                "--mount", "type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix",
            ])
            authority = Path(os.environ.get("XAUTHORITY", str(Path.home() / ".Xauthority")))
            if authority.is_file():
                docker.extend([
                    "--mount", f"type=bind,source={authority},target=/tmp/atlas.xauth,readonly",
                    "-e", "XAUTHORITY=/tmp/atlas.xauth",
                ])
        docker.append("-it" if sys.stdin.isatty() and sys.stdout.isatty() else "-i")
        for variable in ("ATLAS_WHEEL_BACKENDS", "ATLAS_DEFAULT_ENGINE", "CMAKE_CUDA_ARCHITECTURES"):
            if variable in os.environ:
                docker.extend(["-e", variable])
        command = command or ["bash"]
        docker.extend(["--entrypoint", command[0], image, *command[1:]])
    os.execvp("docker", docker)


if __name__ == "__main__":
    main()
