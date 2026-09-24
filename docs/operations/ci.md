# Continuous Integration

Core and Python validation run automatically for pushes to `main` and pull
requests targeting `main`. Both workflows also support manual dispatch. The
workflow definitions describe configured coverage; consult the individual
Actions run before claiming that validation passed.

| Workflow | Trigger | Configured coverage | Artifacts |
|---|---|---|---|
| [`ci-core.yml`](../../.github/workflows/ci-core.yml) | `main` push, `main` pull request, manual, reusable call | Core, Interactive, and rendering geometry tests in the standard TBB development image on Ubuntu 22.04 and 24.04 | CTest failure logs |
| [`ci-python.yml`](../../.github/workflows/ci-python.yml) | `main` push, `main` pull request, manual, reusable call | TBB source distribution and wheel inside the same development image on Ubuntu 22.04/Python 3.10 and Ubuntu 24.04/Python 3.12; metadata, engine selection, Python tests, and the maintained example | Source distributions, wheels, and logs |
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
does not open the native OpenGL window. Core CI uses `tbb-test` to build all native test suites, examples, and the
benchmark target; CTest executes the aggregate Core suite once and both
Interactive suites. Python CI runs `scripts/check_python_package.py` in one
container so its wheel installation and tests share the same environment.

Publication credentials, version preparation, tags, and release ordering are
covered by the [release guide](releases.md).

## Shared build environment

Core and Python CI build `tbb-dev` from the checked-out Dockerfile using Buildx.
The Ubuntu matrix selects the image base, while the Actions host runs Ubuntu
24.04. Neither workflow installs Atlas build libraries directly onto that host.
Both reuse development-image layers and retain ccache with a key including the
Dockerfile, dependency manifest, Python build requirements, and source commit.
Build outputs live under `build/docker-tbb-ubuntu<version>/build` on the runner.

Publication checkout no longer initializes submodules. Python publication builds
an sdist containing the dependency installer and license notices; manylinux's
`before-all` invokes that installer with oneTBB enabled. Docker publication uses
the same development dependencies, then builds separate TBB/CUDA runtimes.
`release.yml` retains its CI gates and publication ordering.

Workflow definitions can be checked locally with `actionlint`. Static validation
does not execute hosted workflows or exercise registry/PyPI permissions. Actual
GPU tests are run locally with the CUDA development image; hosted Actions do not
currently provide that hardware.
