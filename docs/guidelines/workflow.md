# Workflow

The working agreement for contributing to Atlas Engine: how to communicate,
how to scope a change, when to pause and ask, and how to commit.

---

## 1. Communication

- Respond in Korean unless the user asks for another language.
- When a request depends on missing context or an unclear invariant, explain
  the gap instead of guessing or adding unrelated safeguards.

---

## 2. Change Discipline

- Make changes bold enough to fully satisfy the requested behavior. Do not
  preserve broken structure just to keep a diff small.
- Keep edits focused on the requested behavior, but do not treat minimal line
  count as a goal.
- Preserve existing style, naming, include order, file layout, and backend
  portability. Follow the structure recorded in the per-module docs under
  [`docs/atlas/`](../atlas/).
- Do not introduce broad refactors, public API changes, new dependencies,
  build-system changes, or formatting-only churn unless explicitly requested.
- Avoid changes that predictably break builds or leave declarations and
  definitions inconsistent. If the correct fix requires touching related files,
  update them together.
- Prefer project abstractions and existing invariants over ad-hoc workarounds.

---

## 3. When to Be Cautious

Be cautious (prefer a conservative change and confirm intent) only when:

- the user asks for a conservative change,
- public APIs or cross-module contracts would change, or
- the invariant needed for a larger fix is unclear.

Outside these cases, implement the requested behavior directly.

---

## 4. Keeping Docs and Code in Sync

- The per-module docs under [`docs/atlas/`](../atlas/) are the source of truth
  for program structure — every module under `include/atlas/` has one. When you
  make an intentional change to a module, update its document in the same
  change; if the change alters the step pipeline or the framework's shape,
  update the framework overview in [`README.md`](README.md) too.
- When you change a guideline that the code depends on, update the matching
  guideline document here too.

---

## 5. Git and Commits

- When the user asks for `git commit`, group all staged changes except
  `.idea/workspace.xml` by related purpose, then create commits that match those
  groups.
- Commit `.idea/workspace.xml` only when the user asks for `git commit all`, and
  only after all other grouped commits are complete.
