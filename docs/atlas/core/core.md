# Core

The core module is the shared machinery behind the engine's **tagged-union leaf**
pattern. A concrete *umbrella* type (`Collider`, `Sink`, `Material`, `Codec`,
`DsmcKernel`, `Geometry`, `Source`, …) owns a discriminant tag plus an anonymous
`union` of self-contained *leaf* types, and dispatches every operation to
whichever leaf the tag currently names. `DeviceVariant` and `HostVariant` are the
two stateless dispatchers that manage that union — they place-construct, copy or
move, destroy, and visit the active member by switching on the tag — so an
umbrella never hand-writes a `switch` over its leaves. `macros.h` supplies the
`ATLAS_HOST` / `ATLAS_DEVICE` / `ATLAS_ALL_DEVICE` execution-space qualifiers that
decide *where* that dispatch is allowed to run.

The whole reason there are two variants is one property of the parallel backend:
under CUDA a `DeviceBuffer<T>` (`thrust::device_vector<T>`) has a **host-only**
copy constructor. A trivially-copyable umbrella can therefore be byte-copied into
a device lambda and dispatched inside a kernel — that is `DeviceVariant`. An
umbrella whose leaves own a `DeviceBuffer` cannot be copied on the device at all,
so it is held move-only and stays on the host — that is `HostVariant`.

## Files

| File | Role |
|---|---|
| `include/atlas/core/device_variant.h` | `DeviceVariant`, `DeviceVariantCase`, `DeviceTypeSwitch`, `DeviceTypeCase`, `type_tag`, `dependent_false_v`, `member_pointer_traits` — the host+device, copy-based dispatcher for trivially-copyable leaves |
| `include/atlas/core/host_variant.h` | `HostVariant`, `HostVariantCase`, `host_member_pointer_traits`, `host_variant_dependent_false` — the host-only, move-based dispatcher for buffer-owning leaves |
| `include/atlas/core/macros.h` | Execution-space qualifiers (`ATLAS_HOST`, `ATLAS_DEVICE`, `ATLAS_ALL_DEVICE`) and helper macros (`ATLAS_FORCE_INLINE`, `ATLAS_UNROLL`, `ATLAS_NODISCARD`, `ATLAS_MAYBE_UNUSED`, `RESTRICT`, `ATLAS_DEBUG`, `ATLAS_IF_DEBUG`) |

All three are header-only; there is no `src/atlas/core/`.

## The storage contract

Both dispatchers are **stateless** helper templates — they hold no data of their
own. The umbrella class owns the storage and must expose two things by public
name:

- a data member literally named `type`, of the discriminant enum, and
- an anonymous `union` with one member per leaf.

Each *arm* of a variant is a `DeviceVariantCase<TagValue, Member>` (or
`HostVariantCase`) that pairs an enumerator with a **pointer-to-member** into that
union, e.g. `DeviceVariantCase<SinkType::surface, &Sink::surface>`. The dispatcher
recovers the umbrella type and the leaf payload type from the pointer-to-member
via `member_pointer_traits`, so it never needs those spelled out separately.

`DefaultTag` (the third template parameter) is the fallback arm selected whenever
an unrecognized tag is seen. It keeps every dispatch *total*: a corrupt or
default-initialized `type` is folded back to a real arm rather than reading an
inactive union member. The case for `DefaultTag` must therefore always be present
in the list.

## The decision rule

This is the one choice a new umbrella has to get right.

- **Every leaf is trivially copyable and owns no `DeviceBuffer`** → use
  **`DeviceVariant`**. All its methods are `ATLAS_ALL_DEVICE`, so the umbrella can
  be captured by value into a `__host__ __device__` lambda, stored in a
  `DeviceBuffer<Umbrella>`, and dispatched inside a kernel. `DsmcKernel`, `Sink`,
  `Collider`, `Material`, `Codec`, and `Geometry` are all `DeviceVariant`s.
- **Some leaf owns a `DeviceBuffer` (is move-only)** → use **`HostVariant`**. Its
  methods are `ATLAS_HOST` only and the lifecycle is move-based; the umbrella is
  move-only and lives only on the host (typically behind a `…HostPtr`). `Source`
  is a `HostVariant`.

What breaks if you pick wrong:

- **Buffer-owning leaf in a `DeviceVariant`.** `DeviceVariant` copies its leaves
  (`copy_construct`, `assign`, and the `= default` copy members the umbrella
  relies on). `thrust::device_vector`'s copy constructor is host-only, so the copy
  the device-capturable value would require does not exist on the device — the
  umbrella stops being trivially copyable and its `= default` special members
  become ill-formed. It fails to compile, and that compile error *is* the check.
- **Trivially-copyable leaf forced into a `HostVariant`.** It compiles, but you
  have given up the entire point: the umbrella is now move-only and host-only, so
  it can no longer be byte-copied into a device kernel. Anything downstream that
  expected to capture it by value or store it in a `DeviceBuffer` breaks.

The rule of thumb, stated in `buffer.md`: a `DeviceBuffer` is the thing a kernel
may not capture, so any umbrella reachable inside a kernel must be a
`DeviceVariant`, and any leaf that owns a `DeviceBuffer` forces its umbrella to be
a `HostVariant`.

## Operations

`DeviceVariant<Owner, Tag, DefaultTag, Cases...>` exposes:

| Method | Effect |
|---|---|
| `contains(tag)` | `true` iff some arm has that tag. |
| `normalize(tag)` | `tag` if `contains(tag)`, else `DefaultTag` — the total-dispatch guard. |
| `construct(owner, tag, args…)` | Sets `owner.type = normalize(tag)`, then placement-news the matching leaf from `args`. **Assumes no member is currently active** (initialization, not reassignment). |
| `construct_payload(owner, payload)` | Deduces the arm from the leaf's *type* and copies it in; an unsupported type is a `static_assert` failure. |
| `copy_construct(owner, other)` | Copies `other`'s active leaf into an uninitialized `owner`. |
| `assign(owner, other)` | Self-assign is a no-op; otherwise `destroy` then `copy_construct`, so the active tag may change. |
| `destroy(owner)` | Runs the active leaf's destructor; `owner.type` is left unchanged and must not be used to read storage until a leaf is reconstructed. |
| `visit(owner, visitor, fallback)` | Returns `visitor(active leaf)`, or `fallback` if the tag matches no arm. Const leaf. |
| `apply(owner, visitor)` | Invokes `visitor` on the active leaf for a side effect (mutable and const overloads); silently skips if the tag matches no arm. |
| `visit_type(tag, visitor, fallback)` | Dispatches on a tag with *no object in hand*, passing a `type_tag<payload>` so the visitor can recover the leaf type for a size/type query. |

`HostVariant<Owner, Tag, DefaultTag, Cases...>` mirrors that surface —
`contains`, `normalize`, `construct` (perfect-forwarding), `construct_payload`
(matches on the *decayed* payload type), `destroy`, `visit`, and both `apply`
overloads — but **replaces the copy path with a move path**, because its leaves
are move-only:

| Method | Effect |
|---|---|
| `move_construct(owner, other)` | Sets `owner.type = other.type` **verbatim (not normalized)** and move-constructs the matching leaf from `other`'s. |
| `move_assign(owner, other)` | Self-move is a no-op; otherwise `destroy(owner)` then `move_construct`, so the active tag may change. |

There is deliberately no `copy_construct` / `assign` on `HostVariant`: the leaves
are move-only by construction, which is the entire reason this variant exists.

### `DeviceTypeSwitch` — type-only dispatch, no storage

`device_variant.h` also ships `DeviceTypeSwitch<Tag, DefaultTag, Cases...>`, whose
arms are `DeviceTypeCase<Tag, TagValue, Payload>` — a tag paired with a *type*, and
no pointer-to-member and no union. It maps tags to types with no object involved:

- `visit(tag, visitor, fallback)` — dispatch on a runtime tag, handing the visitor
  a `type_tag<payload>`;
- `holds<Payload>` — compile-time predicate: is this type one of the arms?
- `tag_of<Payload>()` — recover the tag of a known type, or `DefaultTag` if absent.

Use it when a caller needs the tag↔type mapping alone (e.g. to size a leaf or pick
an enumerator from a type) without touching a live umbrella object.

## The structural constraint (a real trap)

Forming a pointer-to-member such as `&Sink::surface` requires the umbrella type to
be **complete**. But the umbrella's own constructors call *into* the variant
(`SinkVariant::construct(*this, …)`), and the `using SinkVariant = DeviceVariant<…,
&Sink::surface, …>` alias needs those member pointers. The two therefore cannot be
interleaved. The layout every real umbrella uses — and the one the tests pin down
— is:

1. **Fully define the umbrella class**, but only *declare* the constructors (and
   any member that reaches back into the variant) inside it.
2. **Then** write the `using XVariant = DeviceVariant<…&X::leaf…>` alias, now that
   `X` is complete.
3. **Then** define those constructors and methods out of line, in terms of the
   alias.

`dsmc_kernel.h` does exactly this:

```cpp
class DsmcKernel final {
public:
    DsmcKernelType type = DsmcKernelType::hard_sphere;
    union {
        HardSphereKernel         hard_sphere;
        VariableHardSphereKernel variable_hard_sphere;
        VariableSoftSphereKernel variable_soft_sphere;
    };

    ATLAS_ALL_DEVICE DsmcKernel() noexcept;                       // declared only
    ATLAS_ALL_DEVICE explicit DsmcKernel(DsmcKernelType) noexcept;// declared only
    ATLAS_ALL_DEVICE DsmcKernel(const DsmcKernel&) noexcept = default;
    ATLAS_ALL_DEVICE DsmcKernel& operator=(const DsmcKernel&) noexcept = default;
    ATLAS_ALL_DEVICE ~DsmcKernel() noexcept = default;
    // …visitor-dispatched methods declared only…
};

// Alias comes AFTER the class is complete, so &DsmcKernel::hard_sphere is well-formed:
using DsmcKernelVariant = DeviceVariant<
    DsmcKernel, DsmcKernelType, DsmcKernelType::hard_sphere,
    DeviceVariantCase<DsmcKernelType::hard_sphere,          &DsmcKernel::hard_sphere>,
    DeviceVariantCase<DsmcKernelType::variable_hard_sphere, &DsmcKernel::variable_hard_sphere>,
    DeviceVariantCase<DsmcKernelType::variable_soft_sphere, &DsmcKernel::variable_soft_sphere>>;

// Constructors defined out of line, now that the alias exists:
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel() noexcept {
    DsmcKernelVariant::construct(*this, DsmcKernelType::hard_sphere);
}
```

Putting the alias *before* the class, or defining the constructors in-class, is a
compile error: the member pointer is formed against an incomplete type. A test
file got this wrong and failed to compile, so the ordering is not cosmetic. The
variant machinery is the reason the umbrellas look "inside-out" like this.

## HostVariant hazards

The move-based lifecycle has sharper edges than the copy-based one, because the
leaves own resources. The tests in `tests/atlas/core/host_variant_tests.cpp` pin
these down:

- **`construct` does not destroy the current member.** It assumes no member is
  active and placement-news over the storage. On a `HostVariant`, re-constructing
  over a live buffer-owning leaf **leaks** that leaf's resource. Callers must
  `destroy` first — the test `ConstructNormalizesUnknownTagToDefault` calls
  `ShapeVariant::destroy(shape)` before `ShapeVariant::construct(...)` for exactly
  this reason, whereas the `DeviceVariant` version needs no such `destroy` because
  its leaves own nothing.
- **`move_construct` has no default-tag fallback.** Unlike `DeviceVariant::
  copy_by_tag` (which reconstructs at `DefaultTag` on an unknown tag),
  `move_by_tag` simply constructs nothing if the tag names no arm, leaving
  `owner.type` pointing at an uninitialized member. It also copies `other.type`
  verbatim rather than normalizing it. This is safe only because callers move from
  validly-tagged objects; do not feed it a corrupt tag.
- **The moved-from source keeps its tag but is emptied.** After
  `move_construct`, the source's leaf is in its moved-from state (its owned buffer
  released) and should only be destroyed afterward — never read.
- **Self-move-assign is a guarded no-op.** `move_assign` checks `&owner == &other`
  and returns without destroying or reconstructing, so no double free.
- **Cross-tag assignment destroys the old leaf first.** `move_assign` runs
  `destroy(owner)` then `move_construct`, so assigning a differently-tagged value
  destructs the previous leaf exactly once before installing the new one.

`DeviceVariant`'s copy path (`assign`) is analogously self-assign-guarded and
destroys-then-copies, but because its leaves are trivial there is no resource to
leak — its only comparable hazard is that `construct` likewise assumes an inactive
member.

## Adding a leaf

Adding a leaf to an existing umbrella (see `collider.md`, `sink.h`,
`dsmc_kernel.h` for worked examples):

1. **Write the leaf** satisfying that umbrella's `Concept…` (`ConceptCollider`,
   `ConceptSink`, `ConceptDsmcKernel`, …). For a `DeviceVariant` umbrella it must
   be **trivially copyable and own no `DeviceBuffer`**; for a `HostVariant`
   umbrella it is move-only.
2. **Add its enumerator** to the discriminant enum.
3. **Add the union member** to the umbrella class.
4. **Add the matching case** to the `using XVariant = …<…>` alias:
   `DeviceVariantCase<Enum::new_leaf, &X::new_leaf>` (or `HostVariantCase`).
5. **Add `static_assert(Concept…<NewLeaf>);`** so a missing method is a compile
   error, not a silent gap.

For a *new umbrella* rather than a new leaf, first apply **the decision rule**
above to pick `DeviceVariant` vs `HostVariant`, then follow the **structural
constraint** layout: define the class with only declarations, write the variant
alias after it, and define the constructors out of line.
