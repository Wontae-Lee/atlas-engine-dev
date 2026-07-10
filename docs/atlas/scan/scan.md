# Scan

`scan` is a single free-function family, `atlas::exclusive_scan`, that turns a
range of per-cell or per-particle counts into the base offsets that pack a dense
output array. It is not a leaf, a builder, or a stateful type — just a
policy-dispatched wrapper over the backend's prefix scan. In the step pipeline it
is the *allocate* phase's arithmetic: the DSMC solver scans per-cell candidate
counts into `_candidate_offsets`, and the fluid step scans the per-particle
survivor flags into `_survivor_offsets`, so that a following kernel writes each
element to `offset[i]` without contention.

## Files

| File | Role |
|---|---|
| `include/atlas/scan/exclusive_scan.h` | The whole module: `scan_plus` alias, `detail::host_exclusive_scan`, and the three `exclusive_scan` overloads. No `src/` counterpart — it is header-only templates. |

## The shape

### Policy dispatch and the backend split

```cpp
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt result, T init, BinaryOp binary_op);
```

`exclusive_scan` computes, for each position, the reduction of everything
strictly before it: `result[0] == init`, `result[i] == binary_op(init, in[0..i-1])`.
`result` may alias `first` for an in-place scan. An empty range (`first == last`)
returns `result` unchanged, before any backend call.

The dispatch is compile-time on the `ExecutionPolicy P` template parameter, split
by backend macro:

- **`ATLAS_BACKEND_CUDA` + `P == device`** → `thrust::exclusive_scan(thrust::device, …)`.
  The iterators must be device iterators; this runs on the GPU.
- **`ATLAS_BACKEND_CUDA` + any other `P`** → `detail::host_exclusive_scan`.
- **`ATLAS_BACKEND_TBB` (any `P`)** → `detail::host_exclusive_scan` unconditionally;
  there is no device branch to compile.

`detail::host_exclusive_scan` lowers any fancy iterator to a raw pointer via
`atlas::raw_pointer_cast(&*it)` and calls `std::exclusive_scan`. It is a separate
function specifically so the CUDA device branch never instantiates it with a
device iterator (a device iterator has no host-dereferenceable `&*it`). It
requires contiguous, host-addressable ranges.

The host scan is **sequential** on purpose: the ranges Atlas scans are a few
thousand per-cell counts, and a parallel scan over that size does not repay its
own synchronization. This is a deliberate simplification, documented in the
header.

### `scan_plus` — the defaulted combine op

```cpp
#if defined(ATLAS_BACKEND_CUDA)
template <typename T> using scan_plus = thrust::plus<T>;
#else
template <typename T> using scan_plus = std::plus<T>;
#endif
```

The two behave identically, but only `thrust::plus` carries the
`__host__ __device__` annotations a device scan needs without leaning on
`--expt-relaxed-constexpr`. The defaulting overloads pass this down.

### The two defaulting overloads

```cpp
// (4-arg) default op = scan_plus<value_type>
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt result, T init);

// (3-arg) default op AND default init = value_type{} (0 for arithmetic)
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
OutputIt exclusive_scan(InputIt first, InputIt last, OutputIt result);
```

Both forward to the 5-argument primary. The 4-argument form is the "counts to
offsets" case both live call sites use, always with `init == 0` and
`P == ExecutionPolicy::device`.

## Deliberately absent

- **No parallel host scan.** The host path is a sequential `std::exclusive_scan`.
  For the few-thousand-element count arrays Atlas scans, a parallel host scan's
  synchronization would cost more than it saves. Stated in the header's `@note`.
- **No inclusive scan, no segmented scan, no scan-by-key.** The module exposes
  exactly the one primitive the allocate phase needs.
- **No `serial`/`host` specialization under CUDA.** Any non-`device` policy under
  the CUDA backend simply routes to the host sequential path; there is no
  attempt to run a TBB or thread-pool scan there.

## Not implemented

| What | Where | Evidence | Kind |
|---|---|---|---|
| 3-argument overload (defaults both `init` and `binary_op`) | `exclusive_scan.h:158` | Both call sites — `dsmc_solver.cu:174` and `fluid.cu:70` — pass four arguments (`first, last, result, 0`). A grep of `include src tests benchmarks` for `exclusive_scan` finds no 3-argument call anywhere; the overload is never instantiated. | Declared, never called. A convenience extension point (the pure "prefix sum from zero" case), not dead by design flaw. |
| 5-argument primary called with an explicit `binary_op` | `exclusive_scan.h:99` | No call site passes a `binary_op`; both use `init == 0` and let `scan_plus` default. The primary is still *reached*, but only through the 4-argument overload's forward — never invoked by name with a custom operator. | Accepted but unexercised. The `BinaryOp` parameter and the whole `scan_plus`-vs-custom distinction have no non-addition user in the tree. |
| Non-`device` execution policy | `exclusive_scan.h:108` (the `else`) and the whole host branch | Both call sites pass `ExecutionPolicy::device`. Under the CUDA backend the `else` (host) branch of the primary is therefore never instantiated; `detail::host_exclusive_scan` is only ever reached under the TBB backend, where `device` lowers to it. | Backend-conditional. Not dead — it is the only path under TBB — but on a CUDA build no caller reaches it. |

There is no orphaned type or state in this module; every symbol is either called
or reached through a defaulting forward, except the 3-argument overload above.

## Extending

- **A new combine operation** (e.g. a max-scan): call the 5-argument primary
  directly with your own functor. On a device scan the functor must be
  `__host__ __device__`; model it on `scan_plus`, not a bare `std::` functor,
  or nvcc will reject the device instantiation without `--expt-relaxed-constexpr`.
- **A new execution policy value**: add it to `ExecutionPolicy` in
  `parallel/parallel_for.h`; `exclusive_scan`'s `if constexpr (P == device)`
  leaves every other value on the host path automatically, so no change here is
  needed unless the new policy wants its own backend call.
- There is no `validate()` or `static_assert` in this module. Mis-pairing an
  iterator with a policy (a host pointer under `P == device`, or a device
  iterator on a host path) fails at compile time inside thrust or `std`, not with
  a local diagnostic.
