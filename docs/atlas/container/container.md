# Container

The container module is the engine's two general-purpose storage vocabularies —
one for the **device**, one for the **host** — that sit underneath other modules
rather than at a step in the pipeline:

- `Container<T, N>` is a fixed-size, inline, trivially-copyable array usable in
  device code — Atlas's `std::array` replacement.
- `TypeStore<Base>` is a host-only, type-keyed heterogeneous store — the map
  `Fluid` and `Universe` use to hold their state objects and look them up by type.

Both are header-only; there is no `src/atlas/container/`.

## Files

| File | Role |
|---|---|
| `include/atlas/container/container.h` | `Container<T, N>` + the `Container2`/`Container3`/`Container4` aliases + `TriangleContainer4` |
| `include/atlas/container/type_store.h` | `TypeStore<Base>` — the type-indexed store of owned singletons |

## `Container<T, N>` — an inline, device-capturable array

`Container` is an aggregate that stores exactly `N` elements of `T` **inside the
object** (a public member `data_[N == 0 ? 1 : N]`), with no heap allocation and no
owning pointer. Because the storage is inline and every accessor is
`ATLAS_ALL_DEVICE`, a `Container` of a trivially-copyable `T` is itself trivially
copyable, so it can be captured by value into a device lambda, passed to a kernel,
or embedded in a `DeviceVariant` leaf. `std::array` cannot fill this role because
its members are host-only under nvcc; `Container` supplies device-callable
equivalents. It is `constexpr`-friendly and `ATLAS_FORCE_INLINE` throughout, so it
lowers to raw register/local-array accesses in device code.

### Construction

- **Default** — value-initializes every element to `T{}` (enabled when `N == 0`
  or `T` is default-constructible). For `N == 0` the one reserved storage slot is
  still value-initialized.
- **Variadic** — `Container(args...)` constructs element `i` from argument `i`;
  the overload is enabled by SFINAE only when the argument count equals `N`
  exactly, so there is no partial initialization.

### Access

| Member | Notes |
|---|---|
| `operator[](i)` | Unchecked, mutable and const; **host and device** |
| `at(i)` | Bounds-checked; **host-only** because it throws `std::out_of_range` on `i >= N` |
| `a()`, `b()`, `c()`, `d()` | Named accessors for elements 0..3; each SFINAE-enabled only when `N` is large enough (`a()` needs `N >= 1`, `d()` needs `N >= 4`); host and device |
| `begin()` / `end()` | Iterators; `end()` uses the **logical** size `N`, so for `N == 0` it equals `begin()` even though one slot physically exists |
| `data()` | Raw pointer to storage, host and device |
| `size()` / `empty()` | `static constexpr`; report the logical count `N` (0 length reports `empty()`) |
| `fill(value)` | Copy-assigns `value` into every logical element |
| `swap(other)` | Element-wise swap via ADL `swap` (falls back to `std::swap`), so it works in device code where `std::swap` on the aggregate may be unavailable |

Every access member except `at()` is unchecked and device-callable; only `at()`
validates and throws, which is why it alone is host-only.

### Aliases and `TriangleContainer4`

```cpp
template <typename T> using Container2 = Container<T, 2>;
template <typename T> using Container3 = Container<T, 3>;
template <typename T> using Container4 = Container<T, 4>;

using TriangleContainer4 = Container4<Float3>;
```

`TriangleContainer4` is the concrete workhorse: one triangle packed as four
`Float3` slots — vertices in `a()`/`b()`/`c()` and the outward face normal in
`d()`. It is the element type of a `TriangleMesh`'s device/host triangle buffer;
being a `Container` of trivially-copyable `Float3`, it is itself trivially
copyable and safe to store in a `DeviceBuffer<TriangleContainer4>` and dereference
on the device. The named accessors are the primary way geometry kernels read it.

## `TypeStore<Base>` — a type-keyed heterogeneous store

`TypeStore` maps each concrete derived type to at most one owned instance of it,
using the type itself as the key (`std::type_index` of `typeid(Value)`). It is a
type-indexed set of singletons: there is one stored `Value` per distinct `Value`,
and inserting a second replaces the first. Values are owned through
`std::unique_ptr<Base>` and always retrieved as their concrete type via
`static_cast` — sound because a value is only ever fetched under the exact key it
was stored with.

`Base` must have a virtual destructor (so the owning `unique_ptr<Base>` destroys
derived objects correctly), and every `Value` must derive from `Base` — enforced
by a `static_assert` on every type-parameterized member.

### How `Fluid` and `Universe` use it

Both subsystems keep their per-simulation state objects one-per-type in a
`TypeStore`, so a subsystem can stash and later retrieve its own state type
without a hand-maintained enum or registry:

```cpp
using FluidStateStore    = TypeStore<FluidState>;      // include/atlas/fluid/fluid.h
using UniverseStateStore = TypeStore<UniverseState>;   // include/atlas/universe/universe.h
```

`Fluid` and `Universe` wrap the store's operations behind thin `…_state` methods
that forward to it — `emplace_state<T>()` → `emplace<T>()`, `set_state<T>()` →
`set<T>()`, `state<T>()` → `get<T>()`, `has_state<T>()` → `contains<T>()`,
`remove_state<T>()` → `remove<T>()`. `System::search()`, for instance, looks up
`_fluid->state<FluidPositionState>()` and
`_universe->state<UniverseNumberParticleState>()` to feed the searcher (see
[`searcher`](../searcher/searcher.md), [`universe`](../universe/universe.md)).

### Operations and ownership semantics

| Member | Semantics |
|---|---|
| `emplace<Value>(args...)` | Constructs a `Value` in place from `args` and stores it under `key<Value>()`; replaces any existing `Value` (`insert_or_assign`). Returns a reference to the stored value, valid until it is replaced or removed. |
| `set<Value>(unique_ptr)` | Stores an already-constructed value, taking ownership; replaces any existing `Value`. **Throws `std::invalid_argument` if the pointer is null.** |
| `get<Value>()` | Non-owning `Value*` (const overload → `const Value*`), or `nullptr` if absent — never throws. |
| `contains<Value>()` | `true` iff a `Value` is stored. |
| `remove<Value>()` | Detaches and **returns** ownership as `std::unique_ptr<Value>`, then erases the map entry; `nullptr` if none was present. |
| `reserve` / `size` / `empty` / `clear` | Map bookkeeping; `clear()` destroys every owned value. |
| `begin`/`end`/`cbegin`/`cend` | Iterate `{type_index, unique_ptr<Base>}` pairs in unspecified (hash) order. |

The invariants the tests pin down: `emplace` twice for the same type leaves
`size() == 1` and the second value wins; distinct types are stored independently;
`get`/`remove` return `nullptr` for an absent type rather than throwing; `set(nullptr)`
throws; `remove` both hands back the value and shrinks the store.

The type key is a function-local `static std::type_index` built once per `Value`
from `typeid(Value)`, so every call for the same type returns the same cached key
— consistent across all operations and cheap after the first.

### Host-only and move-only

`TypeStore` is **host-only**: it relies on RTTI, the heap, and
`std::unordered_map`, none device-callable, so no member is `__device__`. It is
**move-only** — copy construction and copy assignment are `= delete`d because the
owned `unique_ptr` elements are not copyable; moving transfers ownership of the
whole map. It is **not thread-safe**; concurrent mutation and lookup must be
externally synchronized.

## Deliberately absent

- **`Container` does no bounds checking except in `at()`.** `operator[]`, the
  named accessors, `begin()`/`end()`, and `data()` are all unchecked so they stay
  device-callable; the throwing `at()` is the only validated, host-only accessor.
- **`TypeStore` allows one value per type, not many.** Re-inserting the same type
  replaces rather than appends; this is the whole point — it models a set of
  per-type singletons, not a multimap.
- **No device-side `TypeStore`.** RTTI and `unordered_map` rule out a device form;
  state that a kernel must read is exposed instead through the trivially-copyable
  `…View` structs the subsystems build from their stores.

## Not implemented

Both types are fully live. `Container` and its aliases back geometry
(`TriangleContainer4` in the mesh buffers) and any inline fixed array in device
code; `TypeStore` backs `FluidStateStore` and `UniverseStateStore` and is
exercised through `Fluid`/`Universe`'s `…_state` methods and its own tests. There
is no declared-but-unproduced, accepted-but-ignored, or stubbed surface in this
module.

## Extending

- **A new fixed-size inline aggregate** is just a `Container<T, N>` (or a new
  `ContainerN` alias). Keep `T` trivially copyable if the container must reach the
  device; the trivial copyability of `Container<T, N>` follows from that of `T`.
- **A new per-subsystem state family** is a new `TypeStore<NewBase>` where
  `NewBase` has a virtual destructor and every leaf derives from it. Wrap it
  behind `…_state` forwarders as `Fluid`/`Universe` do, and remember the
  one-instance-per-type replace semantics when inserting.
- There is no `validate()` umbrella here — correctness is enforced by
  `static_assert`s (`Value` must derive from `Base`) and by the SFINAE on
  `Container`'s constructors and named accessors, all caught at compile time.
</content>
