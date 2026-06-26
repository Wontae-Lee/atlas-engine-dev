# Contributor Guidelines

This directory holds the working guidelines for contributing to Atlas Engine:
how to scope a change, how to write code, and how to build and test. It is the
companion to [`docs/architecture/`](../architecture/), which describes *what the
program is*; the guidelines here describe *how to work on it*.

| Document | Scope |
|---|---|
| [workflow.md](workflow.md) | Working agreement: response language, change discipline, when to ask, and git/commit rules. |
| [coding-style.md](coding-style.md) | How to write code: scope, structure, naming, control flow, state, comments, and portability. |
| [build-and-test.md](build-and-test.md) | Build/test policy, CMake presets and options, and test-authoring rules. |
| [dependencies.md](dependencies.md) | External and in-tree dependencies, and the benchmark reference submodules. |

## Relationship to the Architecture Docs

The guidelines and the architecture docs are meant to be read together:

- For the program structure, runtime objects, the per-step pipeline, the
  backend model, the module map, and structural conventions, see
  [`docs/architecture/`](../architecture/).
- For how to write and ship a change that fits that structure, use this
  directory.

Where a guideline depends on a structural fact (for example the active-prefix
contract or the builder pattern), it links to the relevant architecture
document rather than restating it, so the two stay consistent.
