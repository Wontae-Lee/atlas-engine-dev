# Releases and Publication

Start a release from **Actions → Release → Run workflow** on `main` and enter
the new `X.Y.Z` version. The workflow validates the request, runs Core and
Python CI, bumps and commits release metadata, creates the `vX.Y.Z` tag, and
publishes Python packages, Docker images, and the GitHub Release.

## Release version and metadata

The release version has three authoritative representations and one matching
README citation reference:

- root `project(... VERSION ...)` in [`CMakeLists.txt`](../../CMakeLists.txt),
- `[project].version` in [`pyproject.toml`](../../pyproject.toml),
- `version` in [`CITATION.cff`](../../CITATION.cff),
- the citation sentence in [`README.md`](../../README.md).

The release workflow runs this command only after both CI workflows pass:

```bash
python scripts/bump_version.py X.Y.Z
```

The script accepts only strict `X.Y.Z` versions. It updates all four files,
sets `date-released` to the current date, removes the previous release-specific
`doi:` from `CITATION.cff` and its link from the README, and verifies that all
versions match. The README's concept DOI remains unchanged. Missing or
ambiguous fields cause a failure; a failed write or verification restores the
original files. The script also rejects the already-current version so it
cannot remove that release's DOI by mistake. The workflow commits the result;
the script itself does not commit, tag, push, or publish an artifact.

The CMake Python binding uses `SKBUILD_PROJECT_VERSION` during package builds
and otherwise inherits the root `PROJECT_VERSION`; it has no independent Atlas
version literal. Wheel builds therefore report the package version, while
ordinary CMake builds report the root project version.

The release request must name a version newer than the current metadata. It
must start from the latest `main` commit, and `vX.Y.Z` must not already exist.
Do not restore an older release DOI while preparing a new version because the
new archive DOI does not exist yet.

## Run the release

Select `main`, enter `X.Y.Z`, and start the workflow. Its Core and Python CI
jobs run against the selected commit before any version change. When they pass,
the workflow runs `bump_version.py`, commits the four metadata files, and pushes
the new commit and lightweight `vX.Y.Z` tag together. If `main` moves during
the run, the push fails instead of publishing an older commit. Repository
branch rules must allow the Actions token to push this release commit.

The Python distribution and Docker image jobs then check out the new release
commit, rather than the commit that started the workflow. The workflow uploads
the Python artifacts to PyPI and publishes the checked Docker images to GHCR.
When both finish successfully, it creates a GitHub Release with generated
release notes. Those notes can be edited on GitHub after publication. The
workflow does not contact Zenodo or write a release DOI.

If publication fails after the commit and tag have been pushed, those Git refs
remain. Resolve the failed publisher and complete the remaining publication
manually; do not try to bump the same version again.

## Python publication

[`publish-python.yml`](../../.github/workflows/publish-python.yml) first builds
and validates a source distribution. It then builds TBB manylinux x86_64 wheels
for the configured CPython versions with cibuildwheel, checks their metadata,
and uploads both artifact groups. Its manual `publish` input can publish them
to PyPI through trusted publishing when dispatched from `main`.

Configure PyPI Trusted Publishers for both entry-point workflows:

| Field | Manual publication | Release workflow |
|---|---|---|
| Repository owner | `Wontae-Lee` | `Wontae-Lee` |
| Repository | `atlas-engine-dev` | `atlas-engine-dev` |
| Workflow filename | `publish-python.yml` | `release.yml` |
| Environment | `pypi` | `pypi` |

PyPI [does not currently accept a reusable workflow as a Trusted
Publisher](https://docs.pypi.org/trusted-publishers/troubleshooting/#reusable-workflows-on-github).
The release therefore reuses the build jobs, downloads their artifacts within
the same workflow run, and performs the small upload job directly in
`release.yml` with `id-token: write`.
CUDA wheels are not published by this workflow; local combined TBB/CUDA wheel
construction is documented in the [Python guide](../frontends/python.md).

For a dry run, open **Actions → Publish Python → Run workflow**, select a ref,
and leave **Publish the validated distributions to PyPI** disabled. Manual
publishing is accepted only from `main`; a full release runs through the
manually dispatched `release.yml` after its version check.

## Docker publication

[`publish-docker.yml`](../../.github/workflows/publish-docker.yml) builds four
independent `linux/amd64` runtime images:

```text
TBB  / Ubuntu 22.04
TBB  / Ubuntu 24.04
CUDA / Ubuntu 22.04
CUDA / Ubuntu 24.04
```

Each image is loaded and checked before upload. The checks cover Python package
dependencies, installed engine and version, NumPy and math imports, native
executable linkage, and the maintained simulation example for TBB. CUDA images
are built and inspected without claiming a GPU runtime test.

The registry is `ghcr.io/wontae-lee/atlas-engine-dev`. Tags generated from the
package version, backend, Ubuntu version, and source commit are:

| Tag pattern | Meaning |
|---|---|
| `<version>-<engine>-ubuntu<version>` | Versioned runtime image |
| `<engine>-ubuntu<version>` | Moving backend/Ubuntu alias |
| `<engine>-ubuntu<version>-sha-<commit>` | Source commit identity |
| `tbb`, `cuda` | Ubuntu 22.04 short alias |

The release workflow grants `packages: write` and publishes the already checked
images. Manual dry runs are available through **Actions → Publish Docker** with
its publish input disabled. GHCR package visibility and repository access remain
repository-owner settings; the workflow does not change them.

Runtime commands and build arguments are documented in the
[Docker guide](docker.md).

## Zenodo and citation

The DOI badge uses the concept DOI, which represents all Atlas releases. A
release-specific DOI is issued only after Zenodo archives the GitHub Release.
The release workflow intentionally does not wait for that DOI or rewrite the
existing tag.

After Zenodo creates the archive, record the DOI in the project metadata on
`main` as a separate documentation change when appropriate. The next invocation
of `bump_version.py` removes that release-specific DOI while preparing the next
version. Keep the concept DOI in the README badge and keep version-specific
citation text aligned with `CITATION.cff`.
