# Continuous Integration

Core and Python validation run automatically for pushes to `main` and pull
requests targeting `main`. Both workflows also support manual dispatch. The
workflow definitions describe configured coverage; consult the individual
Actions run before claiming that validation passed.

| Workflow | Trigger | Configured coverage | Artifacts |
|---|---|---|---|
| [`ci-core.yml`](../../.github/workflows/ci-core.yml) | `main` push, `main` pull request, manual, reusable call | GCC/G++ TBB C++ tests on Ubuntu 22.04 and 24.04 | CTest failure logs |
| [`ci-python.yml`](../../.github/workflows/ci-python.yml) | `main` push, `main` pull request, manual, reusable call | TBB source distribution and wheel on Ubuntu 22.04/Python 3.10 and Ubuntu 24.04/Python 3.12; metadata, engine selection, Python tests, and the maintained example | Source distributions, wheels, and logs |
| [`publish-python.yml`](../../.github/workflows/publish-python.yml) | Manual or reusable workflow call | TBB manylinux wheel build and configured cibuildwheel checks | Source distribution and wheels; optional PyPI publication |
| [`publish-docker.yml`](../../.github/workflows/publish-docker.yml) | Manual or reusable workflow call | TBB and CUDA runtime images on Ubuntu 22.04 and 24.04; package, engine, native-linkage, and TBB example checks | Optional GHCR publication and job summary |
| [`release.yml`](../../.github/workflows/release.yml) | Manual dispatch from `main` with a new `X.Y.Z` input | Core and Python CI, metadata bump, release commit and tag, package and image builds, PyPI and GHCR publication, GitHub Release creation | CI logs, PyPI distributions, GHCR images, and a GitHub Release |

Manual publication defaults to a dry run. Setting its `publish` input uploads
only when manually dispatched from `main`; a full release runs through the
verified release workflow. The release workflow uses the selected `main`
commit, verifies that the input is a newer `X.Y.Z` version, and runs the Core
and Python CI workflows. Only after both pass does it bump metadata and push a
new commit and tag. The publishing workflows build from that new commit. The
GitHub Release is created only after PyPI and Docker publication succeed.

Python and C++ validation remain separate. Docker CUDA checks run on a hosted
runner without GPU access: they validate image construction, package loading,
host-side API behavior, and native linkage, not CUDA simulation. Hosted CI also
does not open the native OpenGL window.

Publication credentials, version preparation, tags, and release ordering are
covered by the [release guide](releases.md).
