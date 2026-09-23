# Releases and Publication

Maintainer procedures live here; installation and simulation usage belong in
the root [README.md](../../README.md).

CI coverage and artifacts are documented in [ci.md](ci.md).

## Build or publish Docker images

Open **Actions → Publish Docker → Run workflow**, select `main`, and choose
whether to enable **Publish the checked images to GHCR**. The default is to
build and check the images without uploading them. No image archive is retained
in that mode; each job records the references it built in its summary.

The workflow builds four independent `linux/amd64` images from the checked-out
Dockerfile: `tbb` and `cuda`, each on Ubuntu 22.04 and 24.04. Every image contains
the Python package, selected engine, maintained examples, and native Atlas
executables. It uses recursive submodule checkout, separate build caches, and
two compiler jobs per image. The Dockerfile supplies the CUDA version and GPU
architecture list.

Each image is loaded locally and checked before publication. Every image must
pass Python dependency checks, version and default-engine checks, NumPy import,
basic vector operations, and native executable linkage checks. The TBB image
also runs the bundled DSMC example. When publication is enabled, the workflow
pushes that checked local image without rebuilding it. Matrix jobs publish
independently; a failure in one combination does not roll back another
combination that has already published.

The registry path is `ghcr.io/<owner>/<repository>` in lowercase. For this
repository it is `ghcr.io/wontae-lee/atlas-engine-dev`. The workflow uses the
repository's `GITHUB_TOKEN` with `packages: write`; separate Docker Hub
credentials are not required. See the
[GitHub container publishing guide](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

Tags include the package version from `pyproject.toml`, the engine, and Ubuntu:

| Tag pattern | Example or meaning |
|---|---|
| `<version>-<engine>-ubuntu<version>` | `0.1.0-tbb-ubuntu22.04` |
| `<engine>-ubuntu<version>` | `cuda-ubuntu24.04`, updated on each publication |
| `<engine>-ubuntu<version>-sha-<commit>` | Includes the full source commit SHA |
| `tbb`, `cuda` | Short aliases for the corresponding Ubuntu 22.04 images |

Re-publishing the same package version replaces its version tags. Commit tags
identify source revisions, but dependency downloads can change between rebuilds;
record the registry digest when an exact image must be retained.

New GHCR packages are private by default. After the first publication, set the
package visibility to public if users should pull without authentication.
An existing package must grant this repository Actions access to publish.
See [GHCR access and visibility](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).
This workflow does not change package visibility. User pull/run commands are
in [docker.md](docker.md#published-images).

## Prepare release metadata

The distribution is named `atlas-engine`; the imported package is `atlas`.
[pyproject.toml](../../pyproject.toml) defines the package version, dependencies,
source archive contents, wheel files, and cibuildwheel matrix.

1. Update `project.version` in `pyproject.toml` and the matching `version` in
   [CITATION.cff](../../CITATION.cff). Review the author and license metadata
   together, and update `date-released` for the intended release.
2. Keep citation version, release date, and DOI referring to the same release.
   Remove a previous version's DOI when preparing a new version; add the new
   release DOI once it has been issued. Keep the README citation in sync.
3. Push the intended release changes to `main` before dispatching publication.

Wheel builds pass `SKBUILD_PROJECT_VERSION` to the native extension, so packaged
`atlas.__version__` follows `project.version`. The fallback version in
[bindings/python/atlas/CMakeLists.txt](../../bindings/python/atlas/CMakeLists.txt) applies
to ordinary CMake builds and should also match the release.

Source distributions explicitly include the required submodule contents;
Git internals and local build outputs are excluded. Source builds still require
system TBB and network access for build tools and Abseil. The wheel includes
the project and bundled dependency license files.

## Build or publish Python packages

Open **Actions → Publish Python → Run workflow** and select `main`.
Leave **Publish the validated distributions to PyPI** unchecked to build and
download the `python-sdist` and `python-wheels` artifacts without uploading them.

The workflow builds the source distribution first, then uses that archive to
build TBB manylinux x86_64 wheels for CPython 3.9–3.13. This catches missing
submodule files before publication. It validates metadata and runs the configured
cibuildwheel import/math checks. The full Python test suite remains a separate
run described in [tests/python/README.md](../../tests/python/README.md).

For publication, create the GitHub environment `pypi` and configure a PyPI
Trusted Publisher with these matching values:

| Field | Value |
|---|---|
| Repository owner | `Wontae-Lee` |
| Repository | `atlas-engine-dev` |
| Workflow filename | `publish-python.yml` |
| Environment | `pypi` |

Then dispatch the workflow on `main` with the publish checkbox enabled. The
upload job runs after both artifact jobs succeed. See the
[PyPI Trusted Publisher setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

CUDA wheels are built locally through `scripts/build_wheels.sh`; the current
publication workflow does not upload them. Packaging details, including the
combined wheel and native extension selection, are in [Python frontend](../frontends/python.md).

## Archive on Zenodo

The repository's citation file records version `0.1.0`, release date
`2026-09-14`, and version DOI `10.5281/zenodo.22752179`. The README badge uses
the concept DOI `10.5281/zenodo.22752178` for the collection of releases.

For a new archive:

1. Enable `Wontae-Lee/atlas-engine-dev` in Zenodo's GitHub integration.
2. Review `CITATION.cff`, including authors, known ORCIDs or affiliations,
   release version, and release date.
3. Commit and push that metadata, then publish a GitHub release with a new tag
   containing the intended commit.
4. Wait for Zenodo to process the release and inspect the record's metadata
   and DOI. Adding a citation file alone does not register a DOI.
5. Record the issued version DOI in `CITATION.cff` and the README citation.
   Keep the concept DOI separate from the version DOI.

See the [Zenodo release archiving guide](https://help.zenodo.org/docs/github/archive-software/github-upload/).
There is no `.zenodo.json` in the repository. If one is added later, Zenodo gives
its metadata precedence over `CITATION.cff`; see the
[citation metadata guide](https://help.zenodo.org/docs/github/describe-software/citation-file/).
Zenodo archiving is independent of the manual Python publication workflow.
