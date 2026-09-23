# Plan 2 — Documentation and AGENTS Restructure

## Goal
Reorganize `AGENTS.md` and `docs/` by document responsibility so GPT/developers can quickly locate the correct source of truth and avoid duplicated or stale instructions.

## Target structure
```text
AGENTS.md
docs/
├── README.md
├── contributing/
│   ├── workflow.md
│   ├── coding-style.md
│   ├── build.md
│   ├── testing.md
│   └── dependencies.md
├── architecture/
│   ├── overview.md
│   ├── simulation-pipeline.md
│   └── frontends.md
├── atlas/
│   └── <module>/<module>.md
├── frontends/
│   ├── python.md
│   └── interactive.md
├── operations/
│   ├── docker.md
│   ├── ci.md
│   └── releases.md
└── plans/
    └── <active-plan>.md
```

Keep `docs/atlas/` because it maps naturally to core modules.

## Document responsibilities
- Root `README.md`: end-user entry point; installation, basic use, citation.
- `examples/README.md`: simulation/tutorial usage and adding cases.
- `AGENTS.md`: short router + only stable global invariants.
- `docs/README.md`: developer documentation index/navigation only.
- `docs/contributing/`: how to change the project.
- `docs/architecture/`: cross-module architecture and data/control flow.
- `docs/atlas/`: implementation/extension details for each core module.
- `docs/frontends/`: Python and Interactive internals/boundaries.
- `docs/operations/`: CI, Docker, packaging/release operation.
- `docs/plans/`: temporary implementation plans; remove or archive once incorporated into permanent docs.

## AGENTS.md
Reduce it to:
1. Stable rules that always apply.
2. A task-to-document routing table.
3. Generated/source-of-truth warnings that are easy to miss.

Do not keep detailed CI matrices, release state, full module descriptions, or simulation-pipeline explanations in `AGENTS.md`.

## Architecture docs
### `architecture/overview.md`
Explain only the large-scale model:
- Atlas Core as the computation layer
- Python and Interactive as independent consumers
- Data / policy / driver roles
- major ownership boundaries

### `architecture/simulation-pipeline.md`
Document `System::update()` stages and ordering:
`emit -> search -> allocate -> solve -> advect -> remove`
Include what each stage reads/writes and why ordering matters.

### `architecture/frontends.md`
State dependency boundaries clearly:
- Core depends on neither frontend.
- Python does not route through Interactive.
- Interactive does not depend on Python.
- Frontend-specific serialization/rendering/NumPy logic stays outside Core.

## Core module docs
Use a consistent section pattern where practical:
```text
Purpose
Public Types
Ownership / State
Execution or Data Flow
Host / Device Behavior
Extension Points
How to Add a New Leaf/Type
Related Files
```
Keep module-specific facts in `docs/atlas/<module>/`, not duplicated in guidelines or AGENTS.

## Contributing docs
- `workflow.md`: communication, scope/change discipline, git/commit behavior.
- `coding-style.md`: naming, layout, control flow, comments/Doxygen, portability.
- `build.md`: CMake options/presets, TBB/CUDA selection, compiler/target rules.
- `testing.md`: C++/Python/Interactive tests, smoke tests, benchmarks, when each applies.
- `dependencies.md`: dependency inventory and integration only; remove examples/benchmark tutorial content.

## Frontend docs
### `frontends/python.md`
Binding architecture, engine selection, registration layout, C++/Python type mapping, NumPy boundary, packaging-specific internals.

### `frontends/interactive.md`
`Transport -> Server -> Session -> SystemFactory -> Core`, protocol/config ownership, headless vs rendering boundary, renderer/session access rules.

User-facing run/tutorial content belongs in root/example README files instead.

## Operations docs
- `docker.md`: image/target structure and runtime/development usage.
- `ci.md`: workflows, triggers, coverage, artifacts; source of truth for CI behavior.
- `releases.md`: version metadata, publication, PyPI/GHCR/Zenodo process.

## Plans
Move one-off implementation instructions such as the current interactive refactor plan into `docs/plans/` with a descriptive name. A plan is not permanent architecture documentation. When completed, transfer enduring facts to the relevant permanent docs and remove the obsolete plan.

## Cleanup rules
- One fact should have one primary source of truth.
- Prefer links over repeating long explanations.
- Separate user tutorial, developer rules, architecture, module internals, and operations.
- Update links after moves and remove obsolete `docs/guidelines/` content once migrated.
- Keep documentation synchronized with intentional code/architecture changes.

## Done when
A GPT/developer can answer these immediately:
- "How do I work on the code?" -> `docs/contributing/`
- "How does Atlas fit together?" -> `docs/architecture/`
- "How does this core module work?" -> `docs/atlas/`
- "How do Python/Interactive connect to Core?" -> `docs/frontends/`
- "How are CI/Docker/releases operated?" -> `docs/operations/`
- "What temporary refactor is in progress?" -> `docs/plans/`
