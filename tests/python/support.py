import os
import subprocess
import sys
import textwrap

import numpy as np


def run_python(source, engine=None, timeout=60):
    environment = os.environ.copy()
    environment.pop("ATLAS_DEFAULT_ENGINE", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if engine is not None:
        environment["ATLAS_DEFAULT_ENGINE"] = engine
    return subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", textwrap.dedent(source)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def simple_fluid(buffer_size=5):
    from atlas import Fluid

    positions = np.array(
        [[0.10, 0.10, 0.10], [0.30, 0.10, 0.10], [0.70, 0.70, 0.70]],
        dtype=np.float32,
    )
    velocities = np.array(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0], [0.0, -2.0, 1.0]],
        dtype=np.float32,
    )
    species = np.array([0, 1, 0], dtype=np.uint64)
    return Fluid.from_arrays(
        positions,
        velocities,
        species=species,
        buffer_size=buffer_size,
    )


def simple_universe():
    from atlas import Float3, Universe

    return Universe(Float3(0), Float3(1), 0.5)


def tetrahedron_mesh():
    from atlas import TriangleMesh

    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    indices = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    return TriangleMesh(vertices, indices)
