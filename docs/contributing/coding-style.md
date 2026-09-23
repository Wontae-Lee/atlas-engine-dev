# Coding Style

How to write code in Atlas Engine. These guidelines describe the writing
discipline; the structural conventions that tie code to the architecture
(the tagged-union leaf pattern, builders, backend portability, solver rules)
live in the per-module docs under [`docs/atlas/`](../atlas/) — start with the
framework overview in [architecture overview](../architecture/overview.md) and the pattern itself in
[`atlas/core`](../atlas/core/core.md). Read both before changing core code.

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
  with `src/atlas/**/*.cu` definitions — a couple of host-only modules use
  `.cpp` — header-inline device code, internal
  helpers kept in the module's own namespace next to their owner). Each module
  doc's "Files" section shows the pairing; see
  [`atlas/core`](../atlas/core/core.md).

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
  ownership changes unless requested. See
  [`atlas/solver`](../atlas/solver/solver.md).

---

## 7. Comments

Everything under `include/atlas/` and `src/atlas/` is documented. New code there
carries documentation too — this is the one place the "don't write comments
unless asked" default does not apply. Comments are written in **English**.

### Form

- Multi-line documentation uses `/** ... */`. Not `///` blocks, not `/*! */`.
- A one-line note trailing a data member or an enumerator uses `///<`; a
  one-line note preceding a declaration uses `///`.
- Inside a function body, use `//`.

```cpp
/**
 * @brief Sorts particles into the grid and records each cell's occupancy.
 *
 * Rebuilds every array from scratch; there is no cached-result short circuit,
 * because a stale index fails silently.
 *
 * @param positions      Particle positions. A null state makes the call a no-op.
 * @param number_particle Optional output. Only the DSMC solver and the Knudsen
 *                        codec read it; a null or mis-sized state warns and is skipped.
 * @param particle_count Number of live particles.
 * @throws std::runtime_error when the grid was never configured.
 */
```

### What to document

Every public and protected member function, free function, class, struct, enum
and enumerator, concept, macro, template parameter, and data member.

Use `@brief`, `@param`, `@tparam`, `@return`, `@throws`, `@note`, `@warning`,
`@see`.

### What to say

A `@brief` that restates the function's name is worse than nothing — it costs a
reader a line and teaches them to skip the next one. Document what the signature
cannot show:

- units, valid ranges, and what a null argument or a zero count does;
- who owns the memory, and how long a view's pointers stay valid;
- whether the code runs on the host, on the device, or both;
- **why the shape is what it is** — a `HostVariant` because the leaf owns a
  `DeviceBuffer`; a member public only because nvcc rejects an extended
  `__host__ __device__` lambda in a private member function; a cached `AABB`
  because recomputing it per particle per collider is the inner loop.

Inside a function body, comment the *why*, never the *what*. The step that
deserves a comment is the one a reader would otherwise mistake for a bug: an
algorithm's phase boundary, a host/device transition, a numerical guard, an nvcc
workaround.

### Deliberate simplifications

The engine targets large domains and skips fine physical detail on purpose. When
a simplification is deliberate, say so, or the next reader will "fix" it. The
dropped NTC remainder, `Solid`'s absent properties returning `1.0f`, and the
missing species-id bounds check on kernel reads are all choices, not oversights.
Note that the bounds check *is* kept where the species id indexes an atomic
write, since an out-of-range write corrupts memory rather than just reading junk.

---

## 8. Instrumentation

- Keep temporary logging, counters, assertions, probes, timing code, and
  instrumentation-only fields out of production code unless requested.

---

## 9. Backend Portability

- Prefer the shared abstractions — `DeviceBuffer<T>`, `HostBuffer<T>`,
  `device_shared_ptr<T>`, and `parallel_for<ExecutionPolicy>(...)` — over
  backend-specific code. Keep backend paths behind `ATLAS_BACKEND_CUDA` /
  `ATLAS_BACKEND_TBB` (defined by CMake; exactly one is present), and give
  device-reachable code the right attributes from `core/macros.h`.
- The core ownership model is in the
  [architecture overview](../architecture/overview.md); backend and toolchain
  switches are in [`build.md`](build.md).
