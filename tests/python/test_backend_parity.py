import json
import unittest

import atlas

from support import run_python


PARITY_PROGRAM = """
import json
import numpy as np
import atlas

from atlas import Float3, Fluid, System, Universe

positions = np.array([[0.1, 0.2, 0.3], [0.8, 0.7, 0.6]], dtype=np.float32)
velocities = np.array([[1, 0, 0], [0, -2, 0.5]], dtype=np.float32)
system = System(
    Fluid.from_arrays(positions, velocities, species=np.array([0, 1], dtype=np.uint64)),
    Universe(Float3(-10), Float3(10), 5),
    0.1,
)
for _ in range(5):
    system.update()
result = {
    "engine": atlas.get_default_engine(),
    "position_shape": system.fluid.positions().shape,
    "position_dtype": str(system.fluid.positions().dtype),
    "species_dtype": str(system.fluid.species().dtype),
    "particle_count": system.fluid.particle_count,
    "step": system.step,
    "grid_size": [system.universe.grid_size.x, system.universe.grid_size.y, system.universe.grid_size.z],
    "cell_count": system.universe.cell_count,
    "positions": system.fluid.positions().tolist(),
}
print(json.dumps(result))
"""


class BackendParityTests(unittest.TestCase):
    def test_tbb_and_cuda_match_for_deterministic_motion(self):
        if set(atlas.available_engines()) != {"tbb", "cuda"}:
            self.skipTest("Both TBB and CUDA extensions are required")
        results = {}
        for engine in ("tbb", "cuda"):
            result = run_python(PARITY_PROGRAM, engine, timeout=120)
            if engine == "cuda" and result.returncode != 0 and (
                "cudaErrorNoDevice" in result.stderr
                or "no CUDA-capable device" in result.stderr
                or "CUDA driver" in result.stderr
            ):
                self.skipTest("CUDA extension is installed but CUDA hardware/runtime is unavailable")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            results[engine] = json.loads(result.stdout.strip().splitlines()[-1])

        tbb, cuda = results["tbb"], results["cuda"]
        for key in (
            "position_shape",
            "position_dtype",
            "species_dtype",
            "particle_count",
            "step",
            "grid_size",
            "cell_count",
        ):
            self.assertEqual(tbb[key], cuda[key])
        self.assertEqual(tbb["engine"], "tbb")
        self.assertEqual(cuda["engine"], "cuda")
        for left, right in zip(tbb["positions"], cuda["positions"]):
            for lhs, rhs in zip(left, right):
                self.assertAlmostEqual(lhs, rhs, places=5)
