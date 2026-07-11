# Sync

`Sync` is a single rigid-body pose: the local-to-world transform of one body. It
bundles a translation, an orientation quaternion, and the two cached rotation
matrices derived from that quaternion (forward and its transpose-inverse) so that
transforming a point, direction, or ray in either direction is a plain matrix
multiply with no per-call quaternion conversion. Sync is not itself a step in the
`System::update()` pipeline; it is the transform half of a [`Unit`](../unit/unit.md),
which pairs it with a `Geometry`. Its transforms are exercised *inside* pipeline
steps — the source step maps cached local emission points to world space
(`SurfaceSource`/`VolumeSource`), the remove step maps world positions back to
local for containment tests (`SurfaceSink`/`VolumeSink`), and `Unit::trace` pushes
a world-space ray into a body's local frame before intersecting its geometry.

## Files

| File | Role |
|---|---|
| `include/atlas/sync/sync.h` | The `Sync` value type, its transform methods, and the nested host-side `Builder`; the `SyncHostPtr` / `SyncDevicePtr` aliases. |
| `src/atlas/sync/sync.cu` | Out-of-line definitions of `Sync::builder`, `Builder::with_rigid_pose`, `Builder::validate`, `Builder::build`, and `Builder::make_host_shared`. |

## The type and why it has this shape

`Sync` is **not** a tagged-union module. There is one concrete type, no
`DeviceVariant`/`HostVariant` umbrella and no leaves, because a pose has exactly
one representation. What it shares with the union modules is the reason it can be
captured by value into a device lambda: it is trivially copyable, owns no external
resource, and every accessor is annotated `ATLAS_ALL_DEVICE` (`__host__
__device__`). `Unit` embeds a `Sync` *by value* (`unit.h:402`), and the source
kernels copy the pose out of the unit into a local `Sync` so the device lambda
captures the transform rather than a reference to the host-side unit
(`src/atlas/source/surface_source.cu:89`).

```cpp
class Sync final {
public:
    Float3 translation;                  ///< World-space position of the body origin.
    Quaternion orientation;              ///< Body orientation; assumed unit.
    Float3x3 orientation_matrix;         ///< Cached local-to-world rotation.
    Float3x3 inverse_orientation_matrix; ///< Cached world-to-local rotation (transpose).
};
```

### The matrix cache and its invariant

The two matrices are a cache derived from `orientation`. The invariant is that
after any write to `orientation` you must call `rebuild_matrices()` to restore
consistency. The setters (`set_orientation`, `set_pose`) and the constructor do
this for you; the members are public so a caller that writes `orientation`
directly can rebuild the cache itself — which is exactly what `Unit::rotate`
does (`unit.h:204-206`: it left-multiplies the quaternion, renormalizes, then
calls `_sync.rebuild_matrices()`).

The inverse is stored as the plain transpose. That is only a valid inverse while
the quaternion stays unit, so the whole design rests on the unit-quaternion
assumption. The Builder rejects the zero quaternion but does **not** renormalize
a non-unit one — see *Deliberately absent*.

### Transform API

Every transform comes in two forms: an out-parameter form (`void f(in, out&)`,
no temporary in hot device code) and a value-returning convenience overload that
just calls the out-parameter form. Points get rotate-then-translate; directions
get rotate only (translation-invariant free vectors); rays transform the origin
as a point and the direction as a vector, preserving parameterization. The actual
arithmetic lives in `atlas::rotate_translate` / `rotate_subtract` / `rotate`
(from `math/math.h`), so `Sync` is only the pose bookkeeping around those.

| Method | Direction | Applies |
|---|---|---|
| `sync_to_world(point)` | local → world | `M * p + t` |
| `sync_to_local(point)` | world → local | `Mᵀ * (p - t)` |
| `sync_dir_to_world(dir)` | local → world | `M * d` |
| `sync_dir_to_local(dir)` | world → local | `Mᵀ * d` |
| `sync_to_world(ray)` / `sync_to_local(ray)` | both | origin as point, direction as vector |

### Builder

`Sync::Builder` is the standard project builder: one fluent setter
(`with_rigid_pose`), a private `validate()` that throws, and the terminals
`build()` (value) and `make_host_shared()`. The default staged pose is the
identity (`_translation = 0`, `_orientation = (1,0,0,0)`), so a Sync can be built
with no setter call. `validate()` throws `std::runtime_error` on a non-finite
translation, a non-finite orientation, or the zero quaternion. `make_host_shared`
returns a `SyncHostPtr`; every current caller (all tests and the cylinder
example) uses `make_host_shared` and hands the result to `Unit::Builder::with_sync`,
which dereferences it into a value copy (`unit.h:417`).

## Deliberately absent

| Absence | Why |
|---|---|
| No renormalization of a non-unit quaternion | The Builder rejects only the zero quaternion (`sync.cu:35-41`); a slightly non-unit input is accepted as-is. The transpose-as-inverse identity assumes unit orientation, and repeated `Unit::rotate` steps renormalize themselves (`unit.h:204`), so the pose stays unit in practice without a per-build renormalize. |
| No scale / non-uniform transform | A `Sync` is a rigid pose only — rotation plus translation. There is no scale component, which is why the inverse can be a transpose and every transform preserves length. |
| No `make_host_unique` terminal | Only `make_host_shared` is provided; `Unit` holds poses by shared handle, so the unique-ownership terminal the other modules' builders expose is not needed here. |

## Not implemented

Everything below is **declared but never called** anywhere in `include/`, `src/`,
`tests/`, or `benchmarks/`. None is dead in a harmful sense: they round out an
otherwise-symmetric transform and mutation API. They are extension points, not
oversights — but nothing in the engine reaches them today.

| What | Where | Evidence | Kind |
|---|---|---|---|
| `set_translation` | `sync.h:281` | No call site. `Unit::move` mutates `_sync.translation` directly instead (`unit.h:175`). | Declared, never called (extension point) |
| `set_orientation` | `sync.h:291` | No call site. `Unit::rotate` writes `_sync.orientation` and calls `rebuild_matrices()` directly (`unit.h:204-206`) rather than going through this setter. | Declared, never called (extension point) |
| `set_pose` | `sync.h:303` | No call site anywhere. | Declared, never called (extension point) |
| `sync_dir_to_local` (out-param + value overloads) | `sync.h:164`, `sync.h:243` | No call site. The other three direction/point directions are used (`unit.h:243,247,248`, sink headers, source `.cu`), but world→local *direction* is never needed by any step. | Declared, never called (extension point) |
| `sync_to_world(Ray)` (out-param + value overloads) | `sync.h:179`, `sync.h:255` | No call site. Only the inverse, `sync_to_local(Ray)`, is used — `Unit::trace` pushes a world ray into local space (`unit.h:243`); nothing pushes a local ray out to world. | Declared, never called (extension point) |
| `SyncDevicePtr` alias | `sync.h:375` | The alias is referenced nowhere; `SyncHostPtr` is used throughout (tests, benchmark, `Unit::Builder::with_sync`). A device-side shared handle to a lone pose is never constructed — poses travel into device code by value inside a `Unit`. | Declared, never used (extension point) |

Note the out-parameter point overloads (`sync_to_world(Float3, Float3&)` etc.)
and `sync_dir_to_world`'s out-parameter form are *not* in this list: they have no
external call site, but the value-returning overloads that are called delegate to
them, so they are reached indirectly. `Sync` has no serialization and no observer
of its own — a pose is persisted and restored as part of its owning `Unit`, so
there is no "no-producer serialized state" case here.

## Extending

- **Add a transform** (e.g. transform a plane or an AABB): add the out-parameter
  form plus a value-returning overload that delegates to it, annotate both
  `ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE`, and route the math through the existing
  `math/math.h` free functions so the pose type stays pure bookkeeping. Keep the
  local↔world pair symmetric (both directions) even if only one is needed yet —
  that symmetry is why several methods currently sit unused.
- **Add a builder input** (e.g. accept Euler angles): add a `with_*` setter that
  stages into a private field, extend `validate()` to throw on the new failure
  mode, and construct through the existing `Sync(translation, orientation)`
  constructor so `rebuild_matrices()` runs exactly once. Nothing is caught by a
  `static_assert`; correctness rests on `validate()` throwing and on the
  unit-quaternion assumption holding.
- **Mutating the pose after build**: prefer the existing setters, or write
  `translation` / `orientation` directly and call `rebuild_matrices()` yourself
  after touching `orientation` — the members are public precisely so device code
  (`Unit::rotate`) can do this without a setter.
