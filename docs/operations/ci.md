# Continuous Integration

Core and Python validation run automatically for pushes to `main` and pull
requests targeting `main`. Both workflows also support manual dispatch. The
workflow definitions describe configured coverage; consult the individual
Actions run before claiming that validation passed.

| Workflow | Trigger | Configured coverage | Artifacts |
|---|---|---|---|
| [`ci-core.yml`](../../.github/workflows/ci-core.yml) | `main` push, `main` pull request, manual | GCC/G++ TBB C++ tests on Ubuntu 22.04 and 24.04 | CTest failure logs |
| [`ci-python.yml`](../../.github/workflows/ci-python.yml) | `main` push, `main` pull request, manual | TBB source distribution and wheel on Ubuntu 22.04/Python 3.10 and Ubuntu 24.04/Python 3.12; metadata, engine selection, Python tests, and the maintained example | Source distributions, wheels, and logs |
| [`publish-python.yml`](../../.github/workflows/publish-python.yml) | Manual or reusable workflow call | TBB manylinux wheel build and configured cibuildwheel checks | Source distribution and wheels; optional PyPI publication |
| [`publish-docker.yml`](../../.github/workflows/publish-docker.yml) | Manual or reusable workflow call | TBB and CUDA runtime images on Ubuntu 22.04 and 24.04; package, engine, native-linkage, and TBB example checks | Optional GHCR publication and job summary |
| [`release.yml`](../../.github/workflows/release.yml) | Existing `vX.Y.Z` tag push | Version verification, reusable package and image builds, PyPI upload, and GitHub Release creation | PyPI distributions, GHCR images, and a GitHub Release |

Manual publication defaults to a dry run. Setting its `publish` input uploads
only when manually dispatched from `main`; tag publication runs through the
verified release workflow. The release workflow starts from the pushed tag
commit, verifies strict `vX.Y.Z` syntax and
matching CMake, Python, and citation versions. It reuses the Python workflow to
build distributions, uploads them to PyPI from `release.yml`, and calls the
Docker workflow with publishing enabled. It creates a GitHub Release only after
both PyPI and Docker publication succeed. It never creates or moves a Git tag.

Python and C++ validation remain separate. Docker CUDA checks run on a hosted
runner without GPU access: they validate image construction, package loading,
host-side API behavior, and native linkage, not CUDA simulation. Hosted CI also
does not open the native OpenGL window.

Publication credentials, version preparation, tags, and release ordering are
covered by the [release guide](releases.md).
