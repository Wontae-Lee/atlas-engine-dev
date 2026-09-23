# Continuous Integration

The workflows under `.github/workflows/` use manual dispatch on `main`. Their
configuration describes intended coverage; it is not proof that a run passed.
Consult the corresponding Actions run and artifacts for results.

| Workflow | Configured coverage | Artifacts or publication |
|---|---|---|
| [`tbb.yml`](../../.github/workflows/tbb.yml) | GCC/G++ TBB C++ tests on Ubuntu 22.04 and 24.04 | Failure logs |
| [`python.yml`](../../.github/workflows/python.yml) | TBB source distribution and wheel on Ubuntu 22.04/Python 3.10 and Ubuntu 24.04/Python 3.12; metadata, engine selection, full Python tests, and the maintained Python example | Distributions and logs |
| [`publish-python.yml`](../../.github/workflows/publish-python.yml) | TBB manylinux wheels and configured import/math checks | Source distribution and wheels; optional PyPI upload |
| [`publish-docker.yml`](../../.github/workflows/publish-docker.yml) | TBB and CUDA runtime images on Ubuntu 22.04 and 24.04; dependency, import, math, and native linkage checks; TBB example smoke | Optional GHCR publication and job summary |

Python and C++ validation remain separate. Docker CUDA checks run without a GPU:
they validate package loading and host-side API behavior, not CUDA simulation.
The hosted workflows do not currently run CUDA physics or a native OpenGL window.

Publication workflows require their explicit `publish` input. See
[releases](releases.md) for credentials, metadata, tags, and publication steps.
