# Contributing workflow

Atlas has three maintained consumption paths: native C++, Python, and native
Interactive. A change should include the related API, documentation, and
example updates so readers see one coherent behavior across those paths.

Build the standard Docker development image and use its mounted checkout for
builds and tests. The same image definition supplies Core and Python CI.
See [build](build.md) and [Docker operations](../operations/docker.md).

## Preparing a change

Start with the relevant [architecture](../architecture/overview.md) and
[module](../atlas/) documentation, then compare it with the implementation.
Preserve the dependency direction from frontends to Core and the TBB/CUDA
backend contract. Keep the change focused; when a public API or ownership
contract changes, describe its migration impact in the change summary.

## Documentation and review

Each module under `include/atlas/` has a matching `docs/atlas/<module>/`
document. Update it with intentional behavior or ownership changes. Update the
[simulation pipeline](../architecture/simulation-pipeline.md) when phase order
changes, and [frontend boundaries](../architecture/frontends.md) when ownership
moves between components. The root README serves users; detailed design,
toolchain, CI, and release information belongs in these guides.

Use the [coding style](coding-style.md) for source conventions, the
[build guide](build.md) for supported configurations, and the
[testing guide](testing.md) for suite layout and relevant validation. A review
should distinguish checks configured in CI from checks actually completed for
the change. Keep commits grouped by a coherent purpose when preparing a patch.
