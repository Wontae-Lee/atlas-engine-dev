# Contributor Guidelines

This directory holds the working guidelines for contributing to Atlas Engine:
how to scope a change, how to write code, and how to build and test. It is the
companion to [`docs/atlas/`](../atlas/), which documents the individual
refactored modules; the guidelines here describe *how to work on it*.

| Document | Scope |
|---|---|
| [workflow.md](workflow.md) | Working agreement: response language, change discipline, when to ask, and git/commit rules. |
| [coding-style.md](coding-style.md) | How to write code: scope, structure, naming, control flow, state, comments, and portability. |
| [build-and-test.md](build-and-test.md) | Build/test policy, CMake presets and options, and test-authoring rules. |
| [dependencies.md](dependencies.md) | External and in-tree dependencies, and the benchmark reference submodules. |

## Relationship to the Module Docs

The guidelines and the per-module docs are meant to be read together:

- For the structure of an individual module — the tagged-union leaf pattern, its
  leaves, and how to extend it — see [`docs/atlas/`](../atlas/).
- For how to write and ship a change that fits the project, use this directory.

Where a guideline depends on a structural fact (for example the builder pattern),
it links to the relevant module document rather than restating it, so the two
stay consistent.
