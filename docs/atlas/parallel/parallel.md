# Parallel

`parallel` is the engine's backend-portable parallel-algorithm layer. Every kernel in the
step pipeline — emit, search, allocate, solve, advect, remove — is expressed as a
`parallel_for<ExecutionPolicy::device>` over a particle or cell index, capturing raw
pointers by value into a `[=] ATLAS_ALL_DEVICE(...)` lambda. The header set turns those
call sites into either a Thrust launch (`ATLAS_BACKEND_CUDA`) or a TBB loop
(`ATLAS_BACKEND_TBB`) at compile time, chosen by an `ExecutionPolicy` template argument, so
no caller ever names a backend. It sits underneath everything else: it is a leaf dependency
with no dependencies inside the engine except `core/macros.h` and `memory/raw_pointer_cast.h`.

## Files

| File | Role |
|---|---|
| `include/atlas/parallel/parallel.h` | Umbrella header; pulls in `parallel_for`, `parallel_fill`, `parallel_sort`. Deliberately excludes `atomic.h`. |
| `include/atlas/parallel/parallel_for.h` | `ExecutionPolicy` enum, `is_integral_index`, and the two `parallel_for` overloads (index range, iterator range). |
| `include/atlas/parallel/parallel_fill.h` | `parallel_fill` — writes one value across a range. |
| `include/atlas/parallel/parallel_sort.h` | `host_sort_by_key`, the `detail::` host lowering helpers, and `parallel_sort` / `parallel_sort_by_key`. |
| `include/atlas/parallel/atomic.h` | `atomic_add` — one portable fetch-and-add for host and device. |

There is no `src/atlas/parallel/`; the module is header-only, and every function is
`ATLAS_FORCE_INLINE` because it is a thin compile-time dispatch over a backend primitive.

## `ExecutionPolicy` and the backend split

```cpp
enum class ExecutionPolicy { serial, host, device };
```

The three values name *intent*, not hardware. Each `parallel_*` function is a template on
`ExecutionPolicy P` and branches on it with `if constexpr`, so the unused arms are never
instantiated:

- Under `ATLAS_BACKEND_CUDA`, `device` → `thrust::device` (a real GPU launch), `host` →
  `thrust::host`, `serial` → `thrust::seq`.
- Under `ATLAS_BACKEND_TBB` there is no CUDA toolkit at all: `serial` runs a plain loop /
  `std::` algorithm, and `host` **and** `device` both become a `tbb::parallel_for`. On the
  host backend a `DeviceBuffer` *is* host memory, so `device` sorting/filling on the host is
  correct — the pointers are ordinary host pointers.

Because `device` still resolves to a host-thread run under TBB, every device lambda in the
engine is annotated `ATLAS_ALL_DEVICE` (`__host__ __device__` under CUDA, nothing under TBB)
rather than `__device__`.

## `parallel_for` — the workhorse

Two overloads, disambiguated by whether the first argument is integral (`is_integral_index`
SFINAE):

```cpp
template <ExecutionPolicy P, typename IndexType, typename Function, is_integral_index<IndexType> = 0>
void parallel_for(IndexType start, IndexType end, const Function& func);   // [start, end)

template <ExecutionPolicy P, typename InputIt, typename Function,
          std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
void parallel_for(InputIt first, InputIt last, const Function& func);      // [first, last)
```

The **index overload** is what the engine uses: under CUDA it is `thrust::for_each` over a
`counting_iterator`, so `func(i)` runs per index; under TBB it is a `tbb::parallel_for` over
a `blocked_range`, letting TBB pick the grain size. `end <= start` returns without launching.
`func` runs concurrently, so it must touch only atomics or indices it alone owns.

The **iterator overload** dereferences and passes `func(*it)`; under TBB the range must be
random-access so the `blocked_range` can index `first + i`.

## `parallel_fill`

```cpp
template <ExecutionPolicy P, typename Iterator, typename T>
void parallel_fill(Iterator first, Iterator last, const T& value);
```

`thrust::fill` under CUDA, a TBB `std::fill`-per-block otherwise. Empty range returns early
to avoid a needless kernel launch. Used to clear fluid and universe state.

## `parallel_sort` / `parallel_sort_by_key`

```cpp
template <ExecutionPolicy P, typename RandomIt>
void parallel_sort(RandomIt first, RandomIt last);

template <ExecutionPolicy P, typename KeyIt, typename ValueIt>
void parallel_sort_by_key(KeyIt keys_first, KeyIt keys_last, ValueIt values_first);
```

Only `device` under the CUDA backend takes the GPU path (`thrust::sort` /
`thrust::sort_by_key`). Every other combination lowers the iterators to raw pointers via
`atlas::raw_pointer_cast` and sorts contiguous host memory. The lowering lives in
`detail::host_sort_range` / `detail::host_sort_by_key_range` **specifically so the CUDA
device branch never instantiates a host sort on a device iterator** — a `thrust::device_ptr`
would not survive `raw_pointer_cast(&*it)`.

`host_sort_by_key` sorts an *index permutation* rather than swapping key/value pairs in
place: it `std::iota`s a permutation, sorts it by the key each index points at, snapshots
both arrays, then scatters through the permutation. This needs only a less-than on keys and
never requires the values to be comparable or swappable, at the cost of `O(count)` extra
memory (two full-length copies). Neither `std::sort` nor `tbb::parallel_sort` is stable, so
equal keys may be reordered. The `bool Parallel` template argument (`P != serial`) picks
`tbb::parallel_sort` vs `std::sort`.

## `atomic_add`

```cpp
template <typename T>
ATLAS_ALL_DEVICE T atomic_add(T* address, const T value);   // returns the prior value
```

One portable fetch-and-add. In a `__CUDA_ARCH__` pass it forwards to CUDA `atomicAdd`
(whose overloads decide which types the GPU supports); otherwise it uses
`__atomic_fetch_add` with `__ATOMIC_RELAXED`. It returns the *prior* value, so a caller can
use it as a unique reservation index. Relaxed ordering makes the RMW atomic but imposes no
ordering on surrounding memory, and the return value must not be relied on to synchronize
other memory. It is not aggregated into `parallel.h`; a translation unit that needs it
includes `atlas/parallel/atomic.h` directly. The primitive is currently exercised by its
focused tests and remains available to backend-portable kernels.

## Deliberately absent

- **No reductions, scans, or prefix-sum here.** Prefix sums live in the separate `scan`
  module (`include/atlas/scan/exclusive_scan.h`); this module is intentionally just
  for-each / fill / sort / atomic. A count-and-compact is expressed as a `scan` followed by
  a `parallel_for` scatter, not a fused primitive.
- **No `atomic_min` / `atomic_max` / CAS.** Only fetch-and-add is provided.
- **No stable sort.** Both backends' sorts are unstable and the docs say so; the engine's
  sort keys (cell ids) do not need a tie-break.
- **No grain-size / block-size control.** TBB picks the grain and Thrust picks the launch
  config; the engine does not tune them, consistent with the "easy, fast, large-domain"
  goal.

## Not implemented

Everything in this module compiles, but several declared surfaces are never reached by any
caller in `include/`, `src/`, `tests/`, or `benchmarks/`. None is dead in a broken sense;
each is an unused-but-valid extension point. Recorded, not proposed for deletion.

| What | Where | Evidence | Verdict |
|---|---|---|---|
| `ExecutionPolicy::serial` | `parallel_for.h:28` | No call site anywhere passes `serial`; grep for `ExecutionPolicy::serial` outside `parallel/` returns nothing. Only `device` (32 call sites) and `host` (3, both in `bounding_volume_hierarchy/`) are ever selected. Every dispatch handles `serial` as the `else` arm, but it is never instantiated. | Extension point — the deterministic single-threaded arm exists for debugging/reproducibility but nothing requests it. |
| `parallel_for` iterator-range overload | `parallel_for.h:102-130` | Every external `parallel_for` call passes an integer first argument (`0` or `std::size_t{0}`), selecting the index overload. No caller passes iterators. | Extension point — a valid API the engine never happens to use. |
| `parallel_sort` (non-keyed) | `parallel_sort.h:145-159` | No `parallel_sort<...>` call site outside the module; only the umbrella include appears. The engine only ever needs the keyed variant (sort particle indices by cell id in the searcher). | Declared, never called. |
| `host_sort_by_key` (public, host-pointer) | `parallel_sort.h:45-76` | Never called by name anywhere. It is reachable only *indirectly* through `detail::host_sort_by_key_range`, and only on the non-device path — i.e. on the TBB backend, or a hypothetical `host`/`serial` keyed sort. The one keyed-sort call site uses `device`, so under the CUDA backend it is never instantiated. | Reachable only on the host backend; unreachable under CUDA given current call sites. |
| Keyed sort `host` / `serial` policies | `parallel_sort.h:178-192` | The sole `parallel_sort_by_key` caller (`spatial_hashing_searcher.cu:129`) passes `device`. The `host`/`serial` arms exist but no caller selects them. | Extension point / accepted-but-never-selected. |

The `host` policy *is* genuinely used, but only for BVH construction
(`sah_bvh.cu`, `lbvh.cu`) via `parallel_for<ExecutionPolicy::host>` — never for fill or
sort. So `parallel_fill` and `parallel_sort*` are only ever exercised with `device`.

## Extending

- **A new parallel primitive** (e.g. `parallel_transform`, `atomic_min`): add a header
  beside the others, follow the same shape — a single `ExecutionPolicy P` template, an
  `if constexpr` split on `P` inside `#if defined(ATLAS_BACKEND_CUDA)` / `#else`, an early
  return for the empty range, and `ATLAS_FORCE_INLINE`. Add it to `parallel.h` unless, like
  `atomic_add`, it is device-only and better included on demand.
- **A new `ExecutionPolicy` value**: add the enumerator, then extend the `if constexpr`
  chain in *every* `parallel_*` function under both backends — there is no default arm to
  catch a missing case, so an unhandled policy simply falls into the existing `else`
  (currently the `seq` / `serial` path). There is no `static_assert` guarding exhaustiveness;
  the check is manual.
- **A new atomic type**: nothing in `atomic_add` restricts `T`; correctness is delegated to
  the backend. On the device the CUDA `atomicAdd` overload set decides what compiles; on the
  host `__atomic_fetch_add` requires a lock-free integral/pointer type. A type the backend
  does not support fails to compile at the call site rather than at a `static_assert`.
