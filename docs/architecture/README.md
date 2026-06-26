# Atlas Engine Architecture

This directory documents the program structure of the Atlas particle
simulation engine, with a focus on the header-only core under
`include/atlas/`. It is written for contributors (human or AI) who need to
understand how the modules fit together before changing code.

Atlas is a C++20 particle simulation engine with a header-only core,
selectable TBB or CUDA execution backends, optional serialization, and an
OpenGL visualization layer (Vizkit). The public API is aggregated through a
single generated umbrella header:

```cpp
#include <atlas/atlas.h>
```

## Reading Order

| Document | Scope |
|---|---|
| [01-overview.md](01-overview.md) | What Atlas is, the layered design, and where each layer lives. |
| [02-runtime-objects.md](02-runtime-objects.md) | The runtime objects (`Fluid`, `Universe`, `Source`, `Sink`, `Collider`, `Orchestrator`, `System`) and their ownership. |
| [03-simulation-pipeline.md](03-simulation-pipeline.md) | The per-step control flow, including the orchestrator pipeline stages. |
| [04-backend-portability.md](04-backend-portability.md) | The CUDA/TBB backend model: buffers, `parallel_for`, memory, and portability macros. |
| [05-module-map.md](05-module-map.md) | A directory-by-directory catalog of `include/atlas/`. |
| [06-conventions.md](06-conventions.md) | File layout, builder pattern, active-prefix discipline, and naming rules. |
| [07-extensibility.md](07-extensibility.md) | The extension points and how to add a new case (virtual, DeviceVariant `visit`, tag-only). |

## How This Directory Should Be Used

These documents are the reference description of the intended architecture.
When changing code, align edits with the structure and conventions recorded
here. When an intentional architectural change is made, update the relevant
document in the same change so the description stays accurate.

The documents describe structure and contracts, not exhaustive APIs. For the
authoritative signatures, read the corresponding `.h` declaration files under
`include/atlas/`.

## Related Documents

- Top-level [`README.md`](../../README.md) — user-facing overview and build instructions.
- [`CLAUDE.md`](../../CLAUDE.md) — working agreement and coding standards for AI contributors.
- [`docs/atlas/`](../atlas/) — focused theory/implementation notes for specific components
  (for example, the DSMC collision kernels and the quaternion class).
