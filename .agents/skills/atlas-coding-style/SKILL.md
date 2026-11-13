---
name: atlas-coding-style
description: Apply the owner's common coding style when writing, modifying, reviewing, or documenting first-party Atlas code, including Core, Interactive, all examples, bindings, tests, and tooling. Covers code shape, naming, constructors and builders, numeric types, reuse, comments, and language-specific conventions.
---

# Atlas coding style

Read this skill before writing or reviewing first-party code anywhere in this
repository. It specifies **how to write the code**, not which subsystem owns it.
Use the separate Core, Interactive, and Examples skills for architecture and
integration. Repository execution restrictions still apply.

## Decide which convention applies

The user's explicit instructions take precedence. Use this common style, the
[human coding guide](../../../docs/contributing/coding-style.md), and the effective
[.clang-format](../../../.clang-format) settings together. Follow the nearest
well-matched class for details not specified here. Existing files contain older
patterns and inconsistent formatting; their presence is not a requirement to
repeat them. Do not rewrite unrelated code just to normalize it.

The owner explicitly designated `examples/interactive/02_intake` as the coding
style standard for **all examples**, including C++ and Python. Older example
CLI parsers and helper functions do not override this choice. This example
convention does not remove the CLI, validation, or error handling from a service,
binding, test, or tool that actually needs them.

See [scope and source evidence](references/scope-and-evidence.md) when working
outside the usual C++/CUDA implementation or checking why a rule applies.

## Make the implementation readable at the call site

- Write direct, sequential control flow. Do not split a short, single-use step
  into helpers, lambdas, adapters, or wrapper types just to reduce function length.
  A helper or separate type should have an actual reusable responsibility or a
  substantial algorithmic role, such as a kernel, builder, interaction, or policy.
- Prefer a small change to the existing owner over a new abstraction hierarchy.
  For Unit motion, store the required settings as members; do not reintroduce
  `Unit::AngularOscillation` or separate per-motion classes for simple state.
- Search for the existing operation before implementing it again: mathematical
  constants, `Float3` operations, normalization, `Sync`, `Unit::move/rotate`, mesh
  loading, and array-conversion helpers already exist. Do not unwrap and rebuild
  the same data merely to cross a self-imposed helper boundary.
- Keep related operations at the same public/private level. A new public pose
  operation belongs with existing public pose operations; do not hide only that
  operation in a private helper. Keep genuine implementation details private.
- Use a constructor to establish the requested initial state instead of default
  construction followed immediately by a list of setters. Existing setters still
  serve later mutations. Avoid inventing a Builder for every simple value type.
- Keep guards required by the contract, but do not add repeated validation,
  silent repair, broad catch-and-ignore blocks, debug counters, or progress logs
  merely as scaffolding. Do not remove necessary boundary checks in the name of
  making code short.

## Names and C++ layout

- Types use `PascalCase`; functions, variables, and files use `snake_case`.
  Private state normally uses `_member`; public value/view fields stay plain.
  Name counts `<noun>_count`. Keep names short and specific to their role rather
  than repeating every enclosing type or inventing near-identical helper names.
- Preserve public API spellings. Recognized domain names such as `BVH`, `LBVH`,
  and `AABB` need not be expanded; do not rename existing APIs for visual uniformity.
- Use four spaces, no tabs, attached opening braces, and type-side pointer and
  reference symbols (`Unit*`, `const Unit&`). Put a function's return type on the
  preceding line. Use spaced braced initialization (`Float3 { 0.0f, 0.0f, 0.0f }`).
- The formatter has no fixed column limit. Break arguments, fluent chains, and
  expressions where the logical grouping becomes easier to read, rather than
  imposing an arbitrary 80/100/120-column rule. Align related assignment groups
  without reformatting unrelated sections. Keep existing include groups separate.
- Short guards are acceptable (`if (!response.success) return 1;`). Use blocks
  for multi-step branches. Do not expand or compress every branch mechanically.
- Prefer the class order: constructors/destructor, core operations, setters,
  getters, then private implementation/state, unless the immediate family has a
  more specific layout. Match member declaration and initializer order. For
  multiline constructor initializers, the configured style puts commas at the
  beginning of continuation lines.
- Declare in `.h` and define substantial host operations in the matching `.cu`
  for Core or `.cpp` for ordinary host code. Keep device-reachable inline
  operations in headers. Retain the family's `ATLAS_*` annotations;
  `ATLAS_NODISCARD` precedes execution-space/static/inline specifiers.

## Keep members, constructors, and builders consistent

When adding a setting, carry it through the member declaration, all relevant
constructors, Builder storage, `with_*` setter, validation, `build()` arguments,
and reset behavior. Do not update only the setter or only the builder call.

Use `std::optional<T>` when an unset value means something different from zero
or `false`. Keep absence explicit through construction; do not substitute a
magic numeric value. Make defaults consistent across members, constructor
defaults, Builder state, frontend configuration, and documentation unless a
documented contract intentionally differs.

Settings that callers control, including motion smoothing, belong in state and
construction parameters, not hardcoded branches. In existing builder families,
declare `class Builder;` in the owning class, define `Owner::Builder` in that
header, and implement its host methods alongside the owner. Setters return
`Builder&` for chaining. Respect whether that particular builder consumes/reset
its state or supports repeated const builds; do not impose one behavior on all.

## Use the right numeric type and constant

Use `float` and `f` literals for the engine's existing single-precision geometry,
motion, and physical state. Choose the matching type at the start of an
expression instead of introducing `double` intermediates and casting repeatedly.
Use `const` for values that do not change; it does not imply global scope.

Reuse shared math/physical constants from `include/atlas/math/constants.h` and
add genuinely shared constants there when needed. A local immutable value may
stay local; a user-configurable parameter is not a global math constant.

Counts, indices, IDs, seeds, ABI values, and intentional higher-precision
calculations keep their appropriate types. A required index-to-coordinate or
library-interface conversion may use an explicit cast. Do not turn the request
to remove redundant motion casts into a ban on casts or a repository-wide
conversion of integers and doubles to float.

## Write useful, readable comments

Core code carries English Doxygen. Use `/** ... */` for multiline documentation,
`///` for a short declaration note, and `///<` for a short member/enumerator note.
Explain units, ranges, optional-state meaning, ownership, lifetime, and deliberate
numerical choices rather than merely repeating a function name.

Inside functions, use `//`. For examples, explain each meaningful stage in a
short paragraph above the related block: what the caller is configuring, why
the ordering matters, and what state changes. Keep longer explanations off line
ends; reserve trailing comments for short, unambiguous notes. Outside Core and
the requested example style, add new comments only when requested. Keep existing
useful comments synchronized with behavior and avoid boilerplate narration.

## Standard shape for every example

Use [02_intake/main.cpp](../../../examples/interactive/02_intake/main.cpp) as the
writing model, adapted to the frontend's language and API:

- Put setup, construction, execution, and cleanup directly in the entry point.
  C++ uses `main()`; Python may use `main()` with the normal entry guard. Avoid
  one-off `build_system`, `read_file`, `handle`, or argument-parser helpers.
- Do not add CLI case/step selectors, diagnostic progress/angle output, or
  `try/catch` / `try/except` wrappers to these examples unless requested. Use a
  clearly placed run setting or selected configuration file for the intended
  case. Preserve necessary direct response checks and resource cleanup.
- Initialize objects with meaningful constructors or existing Core builders.
  Make stages readable through grouping and useful comments, not extra wrappers.
- In Interactive, keep physical setup in supported engine JSON fields and
  select among separate configuration files. Use the camera constructor and
  existing mesh-loading path. Do not inject example-only JSON keys or reconstruct
  geometry in the example. C++ and Python retain their native APIs; this style
  does not introduce an Interactive/JSON dependency into Core examples.

Existing C++ and Python examples still contain earlier CLI/helper conventions.
Treat them as code to align when an example-style change is requested, not as
exceptions to this standard or permission for an unrelated mass rewrite.

## Review the edited code

Check the actual changed lines for direct flow, sensible names, consistent
constructors/defaults, redundant conversions or duplicated operations, and
comments that explain intent. Use the language-specific notes in the reference
for bindings, tests, Python, and build tooling. This skill does not authorize
formatters, generators, builds, tests, commits, or pushes; follow the repository
bootstrap and the user's execution instructions.
