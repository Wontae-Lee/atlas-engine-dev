# Coding Style

How to write code in Atlas Engine. These guidelines describe the writing
discipline; the structural conventions that tie code to the architecture
(file pairing, builders, active-prefix, solver rules, backend portability) live
in [`docs/architecture/06-conventions.md`](../architecture/06-conventions.md).
Read both before changing core code.

The guiding principle: write code that an experienced C++ programmer would
immediately recognize, and that makes the algorithmic flow understandable at the
call site.

---

## 1. Scope of an Implementation

- Implement only the requested algorithm or behavior. Do not add capability
  that was not asked for.
- Do not split work into many tiny helpers just to shorten individual
  functions. A readable implementation makes the flow understandable at the call
  site, not hidden behind indirection.
- Keep code concise and direct: avoid unnecessary temporary variables and
  redundant branches.

---

## 2. Structure and Organization

- Prefer patterns that experienced C++ programmers recognize: clear ownership
  boundaries, direct control flow, cohesive classes, paired
  declaration/definition files, and role-named helpers.
- Follow the nearest sibling module's organization before inventing a new
  layout.
- Keep helper types close to the owner they support. Split a helper into its
  own type only when the role is substantial and clearly named — a kernel,
  builder, interaction, or policy.
- Match the file and module conventions (`#pragma once`, `.h` declarations
  with `src/atlas/**/*.cu` definitions, header-inline device code, internal
  helpers kept in the module's own namespace next to their owner). See
  [`06-conventions.md`](../architecture/06-conventions.md).

---

## 3. Naming

- Prefer concise class and function names, but optimize for readable control
  flow over raw name length.
- Keep member function names short and natural when the class context already
  supplies meaning. Prefer `apply_collision` over over-explicit names such as
  `accept_and_scatter_pair` or `particle_index_at_offset`.
- Make each algorithmic step distinguishable by name. Avoid near-duplicate
  names that differ only by a generic suffix or a repeated verb (for example
  `execute_*_trial` vs `execute_*_pair`) when more specific step names would
  make the call flow easier to scan.
- Name a count of things `<noun>_count` (`cell_count`, `particle_count`,
  `species_count`, `property_count`). Do not spell it `number_of_<noun>` or
  `num_of_<noun>` — the `_count` suffix is the single house form, on methods,
  fields, and locals alike.
- Spell a well-known domain acronym directly in the type name rather than
  expanding it: `BVH`/`LBVH`/`SAHBVH`, `AABB`. An all-caps acronym type is
  preferred over a long expanded spelling (`BoundingVolumeHierarchy`) when the
  acronym is what the code, files, and call sites already use everywhere else.
- A public rename is a breaking change. Do not rename public spellings (types,
  methods, headers) unless a rename is explicitly requested.

---

## 4. Class Member Order

- Order member functions as: constructors/destructor, the class's core
  functions, setters, then getters.
- Order member variables by type when no nearer sibling layout is more
  specific: `bool`, integer, floating-point, `HostBuffer<T>`, then
  `DeviceBuffer<T>`.

---

## 5. Control Flow and Robustness

- Do not add defensive checks, fallback paths, ownership guards, recovery
  branches, diagnostic-only state, or debug scaffolding unless requested or
  necessary to preserve an existing local contract.
- Do not silently repair invalid states by resetting, zeroing, clamping,
  skipping required work, or mutating unrelated data — unless that repair is
  part of the requested algorithm.

---

## 6. State Mutation

- A guard branch updates only the state owned by its own algorithmic step.
- Do not mutate shared solver, universe, fluid, or searcher state from guard
  branches. Update only the state owned by the requested step.
- Solver code is performance-sensitive: avoid behavior, memory-layout, or
  ownership changes unless requested. See the solver guidelines in
  [`06-conventions.md`](../architecture/06-conventions.md).

---

## 7. Comments and Instrumentation

- Do not write source-code comments unless the user explicitly asks for them.
- When comments are requested, write them in English, and use Doxygen-style
  comments for public APIs or files that already follow that convention.
- Keep temporary logging, counters, assertions, probes, timing code, and
  instrumentation-only fields out of production code unless requested.

---

## 8. Backend Portability

- Prefer the shared abstractions — `DeviceBuffer<T>`, `HostBuffer<T>`,
  `device_shared_ptr<T>`, and `parallel_for<ExecutionPolicy>(...)` — over
  backend-specific code. Keep backend paths behind `ATLAS_TASKING_CUDA` /
  `ATLAS_TASKING_TBB`, and give device-reachable code the right attributes from
  `core/macros.h`.
- The full backend model is in
  [`04-backend-portability.md`](../architecture/04-backend-portability.md).
