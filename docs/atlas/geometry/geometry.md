# Geometry

The geometry module supplies the concrete shapes the rest of the engine tests
points and rays against: the collider sweeps particles against a shape, the sink
and source classify points inside/on a boundary, and the BVH accelerates
triangle-mesh queries. It follows the **tagged-union leaf** pattern used across
the engine (the same one `collider.h` uses): a single concrete umbrella type,
`Geometry`, wraps one of several self-contained leaf shapes and forwards every
query to it.

## Files

| File | Role |
|---|---|
| `include/atlas/geometry/geometry.h` | `Geometry` umbrella (`DeviceVariant`) + `ConceptGeometry` + the query visitor functors + `GeometryVariant` |
| `include/atlas/geometry/geometry_type.h` | `enum class GeometryType : int` — the discriminant |
| `include/atlas/geometry/box.h` | `Box` leaf (axis-aligned) + `Builder` |
| `include/atlas/geometry/circle.h` | `Circle` leaf (flat disk) + `Builder` |
| `include/atlas/geometry/cylinder.h` | `Cylinder` leaf (z-axis, cap/open) + `Builder` |
| `include/atlas/geometry/plane.h` | `Plane` leaf (infinite half-space) + `Builder` |
| `include/atlas/geometry/polygonal_prism.h` | `PolygonalPrism` leaf (regular n-gon extrusion) + `Builder` |
| `include/atlas/geometry/sphere.h` | `Sphere` leaf + `Builder` |
| `include/atlas/geometry/square.h` | `Square` leaf (flat patch) + `Builder` |
| `include/atlas/geometry/triangle.h` | `Triangle` leaf (single face) + `Builder` |
| `include/atlas/geometry/triangle_mesh.h` | `TriangleMeshView` (the union leaf) + `TriangleMesh` (host owner) + `Builder` |
| `src/atlas/geometry/*.cu` | out-of-line `Builder` implementations; `triangle_mesh.cu` also holds the mesh owner's copy/move, BVH/cache builders, OBJ loader, and view forwarders |

The per-leaf `.cu` files (`box.cu`, `circle.cu`, `cylinder.cu`, `plane.cu`,
`polygonal_prism.cu`, `sphere.cu`, `square.cu`, `triangle.cu`) are small: they contain only the
host-side `Builder` methods (`build`, `make_host_shared`, the `with_*` setters,
`validate`). All query math is `ATLAS_ALL_DEVICE` and lives inline in the
headers so it also compiles for the device.

## Design

`Geometry` is a **`DeviceVariant`** (see `core/device_variant.h`): a `type` tag
plus a `union` of the leaf shapes. Because every leaf is trivially copyable, the
whole `Geometry` is trivially copyable (a `static_assert` in
`geometry_tests.cpp` pins this), so it can live in a `DeviceBuffer<Geometry>`
and be dispatched and copied inside a device lambda. All special members are
`= default` — the compiler-generated copy is the flat byte copy the device
needs. Construction and reassignment route through the `GeometryVariant` alias
(placement-new + tagged visitation) so the tag and the live union member never
disagree.

```cpp
struct Geometry {
    GeometryType type = GeometryType::sphere;   // default/fallback tag
    union { Box box; Circle circle; Cylinder cylinder; Plane plane;
            PolygonalPrism polygonal_prism; Sphere sphere; Square square; Triangle triangle;
            TriangleMeshView triangle_mesh; };

    Geometry();                                  // activates the unit sphere
    template <typename Payload> explicit Geometry(const Payload&);  // deduce tag from leaf

    Float3     closest_point(p)  const;          // every query below → active leaf
    Float3     closest_normal(p) const;
    float      signed_distance(p) const;
    bool       is_inside(p, tolerance = 0) const;
    bool       is_on_surface(p, tolerance = 0) const;
    Float3     centroid() const;
    AABB       bound() const;
    bool       is_valid() const;
    HitSurface trace(Ray) const;
};
```

`GeometryType` is a fixed-underlying-type `int` enum with nine enumerators in
declaration order. `sphere` is the default and the fallback: a
default-constructed `Geometry` is the unit sphere, and any tag not registered
`normalize`s to `sphere`.

### Leaf contract — `ConceptGeometry`

Every leaf must satisfy `ConceptGeometry`, which requires the full read-only
query surface with exact return types:

```cpp
template <typename S>
concept ConceptGeometry = requires(const S s, const Float3 p, const Ray r, float tolerance) {
    { s.closest_point(p) }          -> std::same_as<Float3>;
    { s.closest_normal(p) }         -> std::same_as<Float3>;
    { s.signed_distance(p) }        -> std::same_as<float>;
    { s.is_inside(p, tolerance) }   -> std::same_as<bool>;
    { s.is_on_surface(p, tolerance) }-> std::same_as<bool>;
    { s.centroid() }                -> std::same_as<Float3>;
    { s.bound() }                   -> std::same_as<AABB>;
    { s.is_valid() }                -> std::same_as<bool>;
    { s.trace(r) }                  -> std::same_as<HitSurface>;
};
```

- **`closest_point(p)`** — nearest point on the surface to `p`.
- **`closest_normal(p)`** — outward unit normal at the point nearest `p`. The
  flat leaves (`Plane`, `Circle`, `Square`, `Triangle`) ignore `p` and return
  their stored face normal.
- **`signed_distance(p)`** — distance to the surface, negative inside.
- **`is_inside(p, tolerance)`** — containment test; positive `tolerance` dilates
  the shape, negative erodes it. `tolerance` defaults to `0`.
- **`is_on_surface(p, tolerance)`** — surface-shell test of half-width
  `tolerance`; a negative `tolerance` always returns `false`.
- **`centroid()`** — representative center.
- **`bound()`** — world-space `AABB`.
- **`is_valid()`** — whether the leaf's parameters describe a usable shape.
- **`trace(ray)`** — nearest forward ray/shape hit, as a `HitSurface`
  (`is_intersecting == false` on a miss).

Nine `static_assert(ConceptGeometry<Leaf>);` lines in `geometry.h` check every
leaf. Note the triangle-mesh case is asserted through **`TriangleMeshView`**, not
the owning `TriangleMesh` — see below.

Each member of `Geometry` dispatches through
`GeometryVariant::visit(*this, Functor{}, fallback)`: a plain visitor functor
(`GeometryClosestPoint`, `GeometryTrace`, …) is invoked on whichever leaf `type`
selects. They are functors rather than lambdas because nvcc forbids an extended
`__host__ __device__` lambda in class scope. The last argument is the fallback
returned when `type` matches no registered case — which for a `normalize`d tag
cannot happen, but *can* if the discriminant is corrupted directly.

### Fallbacks for a no-active-leaf tag

`geometry_tests.cpp` (`UnknownActiveTagReturnsQueryFallbacks`) forces a bogus
`type` and pins each fallback:

| Query | Fallback |
|---|---|
| `closest_point` | **the query point `p`** |
| `closest_normal` | `Float3(0, 0, 0)` |
| `signed_distance` | `+infinity` |
| `is_inside` | `false` |
| `is_on_surface` | `false` |
| `centroid` | `Float3(0, 0, 0)` |
| `bound` | default (empty) `AABB` |
| `is_valid` | `false` |
| `trace` | empty `HitSurface` |

> **Stale Doxygen.** `geometry.h`'s `@return` for `closest_point` claims the
> no-active-leaf fallback is `Float3(0,0,0)`. The implementation actually returns
> the query point `p` (and the function's own inline comment agrees: *"Fallback
> returns the query point itself (distance zero)"*). The behavior above — `p` —
> is the real one; only that one `@return` line is wrong.

## The triangle-mesh case is special

Every other leaf is a small value type stored directly in the union. The
triangle mesh is not: a mesh owns buffers (and a BVH), which would make
`Geometry` non-trivially-copyable and un-device-copyable. So the module splits
the mesh into two cooperating types:

- **`TriangleMeshView`** — a trivially-copyable struct of **raw pointers**
  (`vertices`, `indices`, `triangle_count`, plus optional BVH fields
  `bvh_nodes` / `bvh_indices` / `bvh_tris` / `bvh_root`). It owns nothing, every
  query is `ATLAS_ALL_DEVICE`, and it is the actual `triangle_mesh` case of the
  union. This is what `ConceptGeometry` is asserted against.
- **`TriangleMesh`** — the **host-only owner**. It holds the triangle
  `HostBuffer`, lazily builds a SAH BVH and a flat query-vertex/index cache,
  keeps a cached `TriangleMeshView` pointing into those buffers, and hands a
  `Geometry` wrapping that view out via `make_device_geometry_view()`.

**Ownership / lifetime rule.** The view is a set of borrowed pointers into the
owning mesh's buffers. The `Geometry` returned by `make_device_geometry_view()`
aliases those buffers and is valid **only while that mesh lives and is not
rebuilt**. Anything that reallocates the mesh's buffers — `set_triangles`,
`load_from_obj`, or moving the mesh — invalidates a previously handed-out view;
you must fetch a fresh one. This is why `TriangleMesh`'s copy/move are
user-provided (`triangle_mesh.cu`): each one calls `update_view()` so the
cached view repoints at *this* instance's buffers rather than dangling into the
source's. On move, the moved-from mesh is reset to an empty, view-cleared state.

`make_device_geometry_view()` calls `ensure_query_cache()` first, so the flat
soup and view pointers are current before the view is copied into the union.

## The leaves

Each leaf is `final`, trivially copyable, defaults to a sensible unit shape, and
has a host-side `Builder` with `build()` (throws `std::runtime_error` on invalid
parameters), `make_host_shared()`, and `with_*` setters. Per-leaf points worth
knowing:

### `Box` — axis-aligned rectangular solid

Stored as two corners `lower_corner`, `upper_corner`; **faces parallel to the
world axes, no rotation**. Default is the unit cube `[-1,-1,-1]..[1,1,1]`.

- **Validity:** finite corners **and** `upper_corner > lower_corner` on *every*
  axis (strict `>`). A collapsed (zero-volume) box — even one flat on a single
  axis — is therefore **invalid**. (This is a recent change from `>=` to `>`;
  `box_tests.cpp` pins that a point-box and a single-axis slab are both invalid.)
- Corners are stored verbatim, so an inverted pair yields an invalid box.

### `Circle` — flat filled disk

`center`, `normal` (need not be unit; normalized on use, `+z` fallback),
`radius`. A zero-thickness surface: its "solid" queries treat the normal as
picking a front/back side rather than enclosing a volume.

- **Validity:** center, normal, radius all finite; normal nonzero; radius `> 0`.
- **`is_inside` is one-sided.** The back half-space (negative plane side) counts
  as inside; a point in front is inside only within the `tolerance` shell
  (`circle_tests.cpp: IsInsideIsTheBackHalfSpace`).

### `Cylinder` — right circular cylinder, z-axis locked

`center` (axis midpoint), `radius`, `height` (total z-extent), `open`. It is
**always aligned to the world z-axis** — no arbitrary orientation. It spans
`z ∈ [center.z − height/2, center.z + height/2]`.

- **Validity:** `radius > 0` **and** `height > 0`.
- **The `open` flag** switches between a capped solid (`false`) and a bare
  lateral tube with no end caps (`true`), which changes the inside/surface
  classification and the ray hit set. The three-argument constructor
  `Cylinder(center, radius, height)` **forces `open = false`**; to get an open
  tube use `Builder::with_open(true)` or set the public `open` field directly
  (the tests do the latter).

### `Plane` — infinite half-space

Hessian form `normal · p + offset == 0`; `normal` is *assumed* unit for metric
queries but is **not** renormalized by the builder. The half-space
`normal · p + offset <= 0` (the side the normal points away from) is the
"inside".

- **Validity:** normal finite and nonzero, offset finite. Does **not** require
  unit length.
- **Overload footgun:** two 2-argument constructors are distinguished only by
  the *type* of the second parameter — `Plane(const Float3& normal, float offset)`
  vs `Plane(const Float3& point, const Float3& normal)`. Passing the wrong second
  argument type silently selects the other meaning. The builder disambiguates
  with named setters (`with_normal_offset`, `with_point_normal`).

### `PolygonalPrism` — regular n-gon extrusion

`center`, integer `side_count`, circumradius `radius`, and total `height`. The
closed prism is always aligned with the world z-axis and includes both caps.
The first cross-section vertex points along `+x`, fixing the polygon's rotation.

- **Validity:** finite center/radius/height, `side_count >= 3`, `radius > 0`,
  and `height > 0`.
- Surface queries scan the polygon edges, so their cost is linear in
  `side_count`. The signed distance and closest point are exact for the regular
  polygon extrusion; ray tracing clips against every side plane and both caps.
- `is_inside` and `is_on_surface` apply tolerance independently along each face
  normal. This preserves the polygonal corners instead of using the rounded
  Euclidean offset implied by a signed-distance threshold.

### `Sphere` — solid ball

`center`, `radius`. Default is the unit sphere at the origin, and it is the
default/fallback leaf of the whole union.

- **Validity:** `radius > 0`.

### `Square` — flat square patch

`center`, `normal` (normalized on use), `side_length`. Zero-thickness, so its
`is_on_surface` is *identical to* `is_inside` (a slab test).

- **Validity:** center, normal, side length finite; normal nonzero; side `> 0`.
- **Orientation is carried only by `normal`.** The in-plane axes are derived on
  the fly from `atlas::orthonormal_basis(normal)`, so the patch's rotation *about
  its own normal* is not independently controllable — it is whatever the basis
  routine produces.
- `closest_normal` ignores its argument and returns the (normalized) face
  normal.

### `Triangle` — single flat face

`a`, `b`, `c`, plus two normal slots: `normal` (the unit face normal *all
queries read*) and `n` (a companion slot written only by the triangle-mesh / BVH
traversal path; the three-vertex constructor leaves it at its `+z` default). The
three-vertex constructor derives `normal = normalized(cross(b−a, c−a))`, falling
back to the zero vector for a collinear triangle. Zero-thickness, one-sided
`is_inside` like the other planar leaves.

- **Validity:** recomputed from the vertices — `cross(b−a, c−a)` nonzero
  (collinear vertices are invalid). Note it does **not** consult the stored
  `normal`.
- **`trace` does not back-face cull.** A ray approaching from behind still
  registers a hit, and the returned normal is the stored `normal`
  (renormalized, `+x` fallback) **unflipped** toward the ray
  (`triangle_tests.cpp: TraceBackFaceHitsWithUnflippedNormal`).
- **`is_inside`'s degeneracy guard keys off the stored `normal` being zero**, not
  off `is_valid()`. A triangle whose stored `normal` is the zero vector returns
  `false` from `is_inside` regardless of tolerance — which is a *different*
  degeneracy signal than `is_valid()` (the latter recomputes the cross product).

### `TriangleMeshView` — triangle-soup leaf

The device-capturable view described above. It holds two representations at
once: a flat triangle soup (always populated by the owner) and an optional BVH
(`closest_point`, `trace`, and the fast winding number use it when present).
Inside/outside is a **generalized winding number** test (`|winding| > 0.5`),
robust even for non-watertight meshes; `signed_distance` combines the
closest-point distance with that winding sign. Public helpers `solid_angle` and
`winding_number` are exposed too.

- **Validity:** non-null `vertices`/`indices`, positive `triangle_count`, and
  every triangle non-degenerate (nonzero cross product). One collinear triangle
  makes the whole view invalid.
- **Empty-view degradation:** `closest_point` returns `p`, `signed_distance`
  returns `+inf`, `centroid` the origin, `bound` a default AABB,
  `closest_normal` returns `+z`, and the classification queries return `false`.
- **Bounds use the BVH root when accessible.** In particular, a CUDA collider's
  device-side `advance()` refreshes its world bound from device BVH storage,
  without dereferencing the mesh owner's host-only flat query cache. Host-side
  CUDA queries retain the flat-soup scan and return the same enclosing box.
- **Low-level device-query limitation:** `TriangleMeshView::centroid`,
  `is_valid`, and exact `winding_number` still read the flat query cache. On a
  CUDA mesh owned by `TriangleMesh`, that cache is host memory, so these methods
  must not be invoked directly from a device kernel. Their Python bindings call
  them on the host; the Python simulation's device paths use the BVH queries.
- **`trace` needs a built BVH.** Unlike the other queries, `TriangleMeshView::trace`
  traverses the BVH on a CPU build; the brute-force flat-soup fallback is
  compiled in **only for the host side of a CUDA build**
  (`#if defined(ATLAS_BACKEND_CUDA) && !defined(__CUDA_ARCH__)`). Consequently a
  hand-built view that carries only the flat vertex/index arrays and *no* BVH
  reports **no hit** on a TBB build, while a CUDA build's host-side query scans
  those arrays and finds the hit (`triangle_mesh_tests.cpp:
  TraceWithoutABuiltBvhFollowsHostBackend`). A view obtained from a real
  `TriangleMesh` (which builds the BVH) traces correctly. The internal
  `has_bvh()` similarly reports "no BVH" on the host side of a CUDA build so
  host-side non-trace queries fall back to the flat soup while the device sees
  the real tree.

### `TriangleMesh` — host owner (not a union leaf)

Owns `HostBuffer`s (the triangles plus the flat query cache) and a
`shared_ptr`-held BVH (`BVHHostPtr`, defaulting to `SAHBVH`, lazily allocated on
first use). Because it owns only host-shareable resources — not a move-only
`DeviceBuffer` — it is **both copyable and movable**, unlike the move-only
`HostVariant` umbrellas (`Source`, `MaterialDictionary`). Copy shares the BVH
pointer and duplicates the buffers; both special members rebuild the cached view
to point at the destination.

- It has **no `trace()` of its own** — ray traversal lives on the view. Tests
  that trace a mesh go through `mesh.make_device_geometry_view().trace(...)`.
- Its host-side query methods (`closest_point`, `signed_distance`, …) are thin
  forwarders to the cached `_view`.
- `TriangleMeshF`, `TriangleMeshHostPtr`, `TriangleMeshDevicePtr` are provided
  aliases.

## Adding a leaf

Mirrors the collider/material checklist:

1. Write a trivially-copyable leaf `final` class satisfying `ConceptGeometry`
   (all nine queries, each `ATLAS_ALL_DEVICE` with the exact return type; add a
   host `Builder` with validation). It must own no `DeviceBuffer`, or `Geometry`
   stops being trivially copyable and the `= default` special members become
   ill-formed. (If the shape needs owned buffers, follow the triangle-mesh
   pattern: store a trivially-copyable *view* in the union and keep the owner
   host-side.)
2. Add its enumerator to `GeometryType`.
3. Add the union member to `Geometry` and a matching `DeviceVariantCase` to the
   `GeometryVariant` alias binding the new tag to that member.
4. Add `static_assert(ConceptGeometry<NewLeaf>);` in `geometry.h` — this catches
   a drifted signature at compile time rather than at the dispatch site.

No new visitor functor is needed unless you add a new *query* to the shared
surface; adding a query means extending `ConceptGeometry`, every leaf, a visitor
functor, and a `visit`-based member on `Geometry`.

Because leaves are trivially copyable, `DeviceVariant` is the correct umbrella
here (contrast `Source`, which owns a `DeviceBuffer` and needs `HostVariant`).
