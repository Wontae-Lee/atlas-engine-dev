# Spatial

`spatial` holds the geometric primitives the engine uses to reason about *where*
things are and *what a ray touches*: an axis-aligned bounding box, a ray plus its
two hit-result structs, and two bounding-volume-hierarchy builders over a
triangle soup. It is the acceleration layer under the geometry module — a
`TriangleMesh` builds a BVH here and traverses it for ray-trace, closest-point,
and winding-number queries, and the collision broad phase in `System::advect`
uses the AABB slab test to cull colliders before the narrow phase.

Unlike the tagged-union leaf modules, `spatial` has no umbrella type. The BVH
builders share a plain abstract base (`BVH`) and are selected by host code; the
AABB and ray types are self-contained value types callable from host and device.

## Files

| File | Role |
|---|---|
| `include/atlas/spatial/axis_aligned_bounding_box.h` | `AABB` value type + `HitAABB`; free helpers `make_aabb`, `merge_aabb`, `aabb_distance_squared`, `transform_aabb` |
| `include/atlas/spatial/ray.h` | `Ray` value type, `HitSurface`, `ray_plane_distance`, `RayF` alias |
| `include/atlas/spatial/bounding_volume_hierarchy/node.h` | `BVHNode` — the flat-array POD node |
| `include/atlas/spatial/bounding_volume_hierarchy/bvh.h` | `BVH` abstract base + `BvhView` alias, pointer aliases |
| `include/atlas/spatial/bounding_volume_hierarchy/lbvh.h` | `LBVH` — Karras Morton-code linear BVH |
| `include/atlas/spatial/bounding_volume_hierarchy/sah_bvh.h` | `SAHBVH` — binned surface-area-heuristic BVH + `Bin` |
| `src/atlas/spatial/bounding_volume_hierarchy/lbvh.cu` | `LBVH` build, Morton codes, Karras range derivation, bottom-up refit |
| `src/atlas/spatial/bounding_volume_hierarchy/sah_bvh.cu` | `SAHBVH` recursive build, binning, SAH cost, partition |

The AABB and ray headers are inline (`ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE`
throughout); only the two builders carry `.cu` implementations, and both build
entirely on the host and upload the result to the device.

## `AABB` — the axis-aligned bounding box

An `AABB` is just its two extreme corners, `lower_corner` and `upper_corner`; a
point is inside when it lies between them component-wise. Every member is
host/device and force-inlined, so a box can be accumulated inside a device lambda.

### The canonical empty box

The default constructor calls `reset()`, which seeds `lower = +inf` and
`upper = -inf` — an *inverted*, deliberately invalid box:

```cpp
AABB box;                    // lower = +inf, upper = -inf
box.is_valid();              // false — non-finite corners
box.is_empty();              // true  — upper <= lower on every axis
box.merge(some_point);       // first merge snaps both corners onto the point
```

The inverted extremes are what make accumulation correct from nothing: the first
`merge` collapses the box exactly onto the merged geometry, because `cmin`/`cmax`
against `±inf` always take the incoming value. `reset()` returns any box to this
state so a reduction can restart.

`is_valid()` and `is_empty()` are distinct tests, and both matter downstream:

- **`is_valid()`** requires *finite* corners and `lower <= upper` on every axis.
  The default empty box is not valid (its corners are infinite). A flat slab
  (zero extent on one axis) *is* valid — it is finite and correctly ordered.
- **`is_empty()`** uses `upper <= lower` on any axis, so a flat slab or a point
  box counts as empty even though it is valid. A point box (`make_aabb(p)`) is
  therefore empty-but-valid.

Most size queries (`width`, `height`, `depth`, `area`, `length`, `extents`) do
*not* guard emptiness — on an inverted box they return negative or meaningless
values by design; callers guard themselves.

### Merging and growing

```cpp
void merge(const Float3& point);   // grow to include a point
void merge(const AABB&  other);    // grow to the union of two boxes
void expand(float delta);          // inflate every face (negative delta shrinks)
```

`merge_aabb(a, b)` is the non-mutating free-function form (returns a fresh union
box), and `make_aabb(p)` builds the degenerate box collapsed onto a single point.

### `trace(Ray) -> HitAABB` — the slab test

`trace` intersects the ray against the three axis-aligned slabs and keeps the
running `[enter, exit]` overlap in the ray parameter `t`:

```cpp
struct HitAABB {
    bool  is_intersecting = false;
    float enter = 0.0f;                                   // t at entry (0 if inside)
    float exit  = std::numeric_limits<float>::max();      // t at exit
};

HitAABB hit = box.trace(ray);
```

Details that the tests pin down:

- `t_enter` starts at `0`, so intersections strictly *behind* the origin are
  excluded — this is a forward query.
- An axis whose direction component is within `eps` of zero is **parallel** to
  that slab: it is a miss unless the origin already lies between the two planes,
  in which case the axis imposes no constraint.
- If the ray *starts inside* the box, `enter` is clamped to `0` so the interval
  begins at the origin, not behind it.
- On a miss `is_intersecting` is `false` and the interval fields hold their
  partially-updated sentinels — read them only when the flag is true.

`intersects(ray)` is a thin boolean wrapper over `trace`.

### Why an unbounded geometry has no valid bound

A plane is infinite, so it has no finite world AABB — its bound stays at the
default empty (invalid) box. This is exactly what the collision broad phase in
`System::advect` keys on: it only runs the slab test when `bound().is_valid()`
is true, and otherwise falls straight through to the narrow phase.

```cpp
// src/atlas/system/system.cu (advect broad phase)
const AABB& bound = colliders[c].bound();
if (bound.is_valid()) {
    const HitAABB coarse = bound.trace(sweep);
    if (!coarse.is_intersecting || coarse.enter > sweep_length) {
        continue;                 // culled: the sweep never reaches this collider
    }
}
// unbounded (plane) colliders skip the cull and always reach the narrow phase
```

Skipping the broad phase for an invalid bound is the correct behavior: a plane
must not be culled just because it cannot produce a box.

### Other helpers

`center`, `extents`, `diagonal_length[_squared]`, `contains(point)` (boundary
inclusive), `overlaps(other)` (touching faces count as overlap, via a strict
separation test), `corner(idx)` (3-bit corner selector), `clamp(point)` (nearest
point on/inside the box), and `aabb_distance_squared(box, point)` (zero inside,
squared to avoid a sqrt in nearest-primitive traversals).

`transform_aabb(bound, fn)` rebuilds a box after an arbitrary point transform by
transforming all eight corners and re-merging them — a conservative axis-aligned
bound of the transformed shape, not the exact rotated box. An invalid input
yields the empty box.

## `Ray`, `HitSurface`, `HitAABB`

### `Ray`

A ray is an `origin` plus a unit-length `direction`. The invariant is enforced
at construction: the two-argument constructor normalizes its input via
`normalized_or`, so the ray parameter `t` is a world-space distance. A zero-length
direction cannot be normalized and falls back to the zero vector — a degenerate
ray every intersection helper treats as "never hits." The default ray sits at the
origin pointing along `+x`.

```cpp
Ray  r(origin, direction);          // direction is normalized on the way in
Float3 p = r.point_at(t);           // origin + t * direction
```

`RayF` is a convenience alias for the single-precision `Ray`.

### `HitSurface`

The result of a ray/surface intersection, returned by value from tracing helpers
(triangle trace, mesh trace, collider trace):

```cpp
struct HitSurface {
    bool   is_intersecting = false;                          // check this first
    float  distance = std::numeric_limits<float>::max();     // t at the hit
    Float3 point    = Float3(0.0f, 0.0f, 0.0f);              // world hit point
    Float3 normal   = Float3(0.0f, 0.0f, 1.0f);             // surface normal
};
```

When `is_intersecting` is false the other members are sentinels and must not be
read. `distance` defaults to `+max` so a "no hit" always loses a nearest-hit
comparison.

### `HitAABB`

The ray/box result described under `AABB::trace`: the `[enter, exit]` interval in
`t`, with `enter` clamped to `0` when the origin is inside.

### `ray_plane_distance`

A free function giving the signed *forward* distance from a ray origin to an
infinite plane. A ray parallel to the plane (denominator within `eps` of zero) is
a miss, and an intersection behind the origin is rejected. On a hit the
out-parameter `distance` is filled; on a miss it is left untouched.

## `BVHNode` — the flat node

Both builders populate arrays of `BVHNode` on the host and upload them verbatim,
so it is a POD with integer child links rather than pointers:

```cpp
struct BVHNode {
    AABB   bounds;                                  // subtree / leaf-primitive bound
    Float3 solid_angle_moment      = {0,0,0};       // area-weighted centroid sum
    Float3 solid_angle_normal_area = {0,0,0};       // area-scaled normal sum
    float  solid_angle_area        = 0.0f;          // total triangle area
    int    left  = -1;                              // child; -1 on a leaf
    int    right = -1;                              // child; -1 on a leaf
    int    start = -1;                              // first index into the reordered array
    int    count = 0;                               // primitives in this leaf
    bool   is_leaf = false;
};
```

A node is either **internal** (`left`/`right` link children, `is_leaf` false,
`start = -1`, `count = 0`) or a **leaf** (`start`/`count` name a contiguous run of
primitives, `is_leaf` true, `left = right = -1`). A default-constructed node is an
interior node with null links and an *empty (invalid)* `bounds`.

`start`/`count` index into the BVH's **reordered `indices` array**, not the
original triangle array; a leaf's true triangle id is `indices[start + k]`.

### The solid-angle moments

The three `solid_angle_*` fields summarize the emissive geometry of a node's
whole subtree for winding-number / solid-angle (fast-winding-number) queries on a
mesh — so a traversal can approximate a distant subtree by its aggregate instead
of descending into it. They are accumulated bottom-up:

- **`solid_angle_moment`** — the area-weighted sum of triangle centroids (the
  first moment); each leaf contributes `(a + b + c) * (area / 3)`. Dividing by
  `solid_angle_area` recovers the area-weighted centroid.
- **`solid_angle_normal_area`** — the sum of area-scaled normals, each
  `0.5 * cross(b - a, c - a)`; its magnitude and direction summarize the
  subtree's net orientation and area.
- **`solid_angle_area`** — total triangle surface area in the subtree.

Degenerate (zero-area) triangles contribute nothing; the NaN-safe `!(area > 0)`
guard also rejects a NaN area. Internal nodes merge their children's moments by
simple summation (`merge_solid_angle_moment`).

## `BVH` — the abstract base

`BVH` is a minimal polymorphic interface with exactly two host operations:

```cpp
class BVH {
public:
    virtual void    build(const HostBuffer<TriangleContainer4>& triangles) = 0;
    virtual BvhView view() const = 0;
};
```

- **`build(triangles)`** replaces any previous state and uploads the result to
  the device. An empty input leaves the hierarchy empty (`root = -1`).
- **`view()`** returns a `BvhView` of raw device pointers for traversal.

`BVHHostPtr` / `BVHDevicePtr` are the shared-ownership aliases, so the geometry
layer can select a build strategy at runtime.

### There is no ray query on the base

`BVH` deliberately exposes **no traversal method**. A BVH does not even define its
own view type: `BvhView` is an alias for `TriangleMeshView`, whose `bvh_*`
members (`bvh_nodes`, `bvh_indices`, `bvh_tris`, `bvh_root`) are exactly the raw
device pointers a device-side traversal needs, gathered into one trivially-copyable
struct a device lambda can capture by value. Both builders' `view()` do nothing
but strip the device-buffer wrappers down to those raw pointers.

Traversal itself lives in **`TriangleMeshView::trace`** (in
`include/atlas/geometry/triangle_mesh.h`), which walks the node array with an
explicit 64-entry stack, culls a node when `nd.bounds.trace(ray)` misses or its
`enter` already exceeds the best hit, and tests leaf triangles with
`Triangle::trace`. On the host side of a CUDA build the BVH pointers are device
memory, so `trace` falls back to a brute-force scan over the flat soup. The
`spatial` tests reproduce this same traversal shape in host code (a recursive
`traverse` pruning on `node.bounds.intersects(ray)`) and check it against a
brute-force loop.

## `LBVH` — Karras Morton-code linear BVH

`LBVH` builds a linear BVH by Karras' (2012) parallel Morton-code algorithm. It
trades tree quality for build speed: it does *not* minimize a surface-area cost
the way `SAHBVH` does, but the per-node work has no cross-iteration dependency and
maps directly onto a device parallel pass (this `.cu` runs the same computation
serially on the host).

### Morton codes and the packed key

`build` computes per-triangle bounds and centroids, then merges all centroids
into a single `centroid_bounds`. Morton codes are quantized against *that*
centroid bound, not the world extent, so the full `[0, 1]` code range is used
regardless of scene scale (`morton3` normalizes each centroid into the centroid
box, quantizes to `_morton_bits` per axis, and interleaves via `expand_bits`).

Triangles are then sorted by Morton code (original index as a stable tiebreak) so
spatially near triangles become array-adjacent. The sorted array stores a **64-bit
packed key** per leaf:

```cpp
keys_sorted[i] = (uint64_t(morton_sorted[i]) << 32) | uint32_t(i);
```

Packing the sorted position into the low 32 bits guarantees *strictly distinct*
keys. This matters because Karras' delta metric (`delta_lcp = clz64(key[i] ^
key[j])`) assumes each leaf owns a distinct key: two triangles with identical
Morton codes — a real possibility at finite quantization — would otherwise give
the same LCP for every pair among them and break the range derivation. The index
tiebreak only affects the LCP once the full 32-bit Morton code already matches.

### Per-internal-node range derivation

Karras' core insight is that internal node `i`'s owned leaf range can be computed
independently of every other node, purely from `i` and the global sorted-key
array — no recursion, no dependency chain. For each internal node the build:

1. Picks a **direction** `d` by comparing the LCP with the left vs. right
   neighbor (grow toward the more similar side).
2. Computes `delta_min`, the LCP any leaf must share with `i` to belong to the
   range at all.
3. **Exponential (doubling) search** for the far end of the range —
   `O(log range)` instead of a linear scan.
4. **Binary search** within `[0, lmax]` to pin the exact range extent.
5. `find_split` locates where the range divides into left/right children — a
   second, independent LCP-boundary search, this time over the range's Morton
   codes (falling back to the midpoint when the endpoints' codes are equal).

### The `2n - 1` layout

For `n` primitives the tree is exactly `2n - 1` nodes: `n` leaves at indices
`[n - 1, 2n - 1)` and `n - 1` internal nodes at `[0, n - 1)`, with **internal node
0 as the root**. `leaf_node_index(k, n) = (n - 1) + k`. A single primitive is a
special case: the lone leaf is the root and the Karras loops are skipped entirely.

### Exactly one primitive per leaf

**Karras' layout stores exactly one primitive per leaf.** The `2n - 1` node count,
the `(n - 1) + k` leaf indexing, and the delta metric's assumption that each leaf
owns one distinct key all depend on it, and the tests assert every leaf has
`count == 1`. Leaf capacity is therefore **not configurable** on `LBVH` — a
`leaf_size` setter was removed for exactly this reason. Packing several primitives
per leaf would require a separate subtree-collapse pass over the built tree, which
this class does not perform. (Contrast `SAHBVH`, whose top-down build recurses
until a leaf is small enough and so exposes `leaf_size`.)

### Bottom-up refit — and why a descending sweep is wrong

Leaf bounds and moments are set directly from each leaf's single triangle. Internal
nodes are then refitted **bottom-up, walking upward from the leaves via parent
links**. Each internal node is refitted by whichever of its two children reaches it
*second*, which guarantees both children are already final when the parent merges
them:

```cpp
HostBuffer<int> visits(n - 1, 0);
for (int k = 0; k < n; ++k) {
    int node = parent[leaf_node_index(k, n)];
    while (node != -1 && ++visits[node] == 2) {   // second child finalizes the parent
        BVHNode& in = h_nodes[node];
        in.bounds = h_nodes[in.left].bounds;
        in.bounds.merge(h_nodes[in.right].bounds);
        merge_solid_angle_moment(in, h_nodes[in.left], h_nodes[in.right]);
        node = parent[node];
    }
}
```

This must be an *upward walk* and **cannot** be replaced by a descending index
sweep. The trap: an internal node whose range *ends* at its own index (direction
`d == -1`) owns a child stored at a *smaller* index, so a parent can precede its
child in the array. A descending sweep assuming array order is bottom-up would
therefore refit some parents before their children — the exact bug that was just
fixed, which left some nodes with empty `inf/-inf` bounds and silently pruned
whole subtrees during traversal (a node whose bound never intersects any ray is
never entered).

The `visits` counter is a plain increment here because the build is host-serial;
on device it becomes an atomic and the same walk parallelizes over the leaves.

### Configuration

`set_morton_bits(int)` clamps into `[1, 10]` (default 10). Ten bits per axis is
the ceiling because three 10-bit axes pack into the 30 usable bits of a 32-bit
Morton code; more bits distinguish nearby centroids more finely at no extra
storage cost. The tests confirm coarse vs. fine quantization can produce a
different leaf ordering.

## `SAHBVH` — binned surface-area-heuristic BVH

`SAHBVH` builds top-down by the binned surface-area heuristic. It produces a tree
that traverses more cheaply than an `LBVH`, at the cost of a slower, serial-
recursive host build. Plainly: **LBVH trades tree quality for build speed; SAH
does the opposite.** `TriangleMesh` defaults to `SAHBVH`
(`src/atlas/geometry/triangle_mesh.cu`).

### The recursive split

`build_recursive` on a range `[start, end)`:

1. Computes the node bound and the **centroid bound**.
2. **Stops** (emits a leaf) when the range fits in one leaf (`count <=
   _leaf_size`), when the centroids are spatially degenerate (extent `<= eps` on
   every axis), or when the chosen axis has zero centroid span.
3. Otherwise bins every centroid into `_bin_count` equal-width buckets along the
   axis of greatest centroid spread (`ext.major_axis()`). Each `Bin` accumulates
   the union bound and count of its primitives.
4. Prefix/suffix sweeps turn per-bin bounds/counts into "everything left of
   boundary `s`" / "everything right of `s + 1`", so every candidate boundary is
   evaluated in `O(1)`.
5. Evaluates the SAH cost at each boundary and keeps the cheapest:

   ```
   C = 1 + (A_left * N_left + A_right * N_right) / A_parent
   ```

   with the traversal constant folded into the `1.0f` term and per-primitive
   intersection cost implicitly 1. A boundary with all primitives on one side is
   skipped — it is not a real split.
6. `stable_partition`s `h_indices[start, end)` so primitives in bins `<=
   best_split` come first, turning the split decision into a concrete contiguous
   partition, then recurses into both halves and merges their bounds and moments.

If no boundary yields a two-sided split (or the partition still lands everything
on one side — a defensive check, since binning only approximates centroid
position), the range falls back to a leaf rather than recursing into a zero-size
child.

### Node storage

The node array is over-allocated to the worst case (`2n - 1`, all-singleton
leaves) up front and trimmed to the count the recursion actually produced. Unlike
`LBVH`, the node count is not structural — a larger `leaf_size` collapses more
primitives into each leaf and yields fewer nodes (a `leaf_size >= n` produces a
single-leaf root).

### Configuration

- **`set_leaf_size(int)`** — clamped to at least 1 (default 32). A range this
  size or smaller becomes a leaf without further splitting.
- **`set_bin_count(int)`** — clamped into `[4, 256]` (default 100). More bins
  evaluate more candidate split planes (finer, potentially better splits) at
  higher per-node cost; the tests confirm correctness holds across bin counts.

## Why the shape is what it is

- **No umbrella, just a base class.** The two BVH builders are selected by host
  code and owned through a `BVHHostPtr`; there is nothing per-element to dispatch
  on at the device level, so there is no `DeviceVariant` here — only the abstract
  `BVH` interface and its two concrete implementations.
- **The view is the mesh view.** A BVH reuses `TriangleMeshView` rather than
  defining its own view, because the raw device pointers a traversal needs are
  already exactly the mesh view's `bvh_*` fields. Gathering them into one
  trivially-copyable struct is what lets a device lambda capture the whole
  acceleration structure by value.
- **Two builders, opposite trade-offs.** `LBVH` is the fast, cross-iteration-free
  Morton build that ports to the device; `SAHBVH` is the higher-quality serial
  recursive build the mesh defaults to. Keeping both behind one interface lets the
  geometry layer pick per use.
- **Host-only build, device traversal.** Both `.cu` files build on the host and
  upload; the traversal that consumes the result is device-side in the geometry
  module. The build code is written free of cross-iteration dependencies (LBVH)
  or as a plain recursion (SAH) precisely at that host/device seam.

## Extending

- **Add a BVH strategy.** Subclass `BVH`, implement `build` (fill `h_nodes` /
  `h_indices`, upload to `d_*`, set `_root`) and `view` (strip device buffers to
  raw pointers into a `BvhView`). Match the existing invariants the tests check:
  `root() == -1` on an empty build, leaves carry `count >= 1` with in-range
  `start`, interior nodes enclose both children, and the leaf primitives form a
  clean permutation of `[0, n)`. Populate the solid-angle moments bottom-up if the
  tree is to support winding-number queries.
- **Do not add a `leaf_size` to `LBVH`.** Its one-primitive-per-leaf invariant is
  structural (see above). Multi-primitive leaves need a subtree-collapse pass that
  does not exist; add that pass first if it is ever wanted.
- **Keep degenerate inputs safe.** Follow the existing short-circuit style — an
  empty triangle soup leaves the hierarchy empty, a single primitive is a leaf
  root, zero-area triangles contribute no moment, and a degenerate centroid
  distribution stops recursion rather than forcing a bad split. These paths are
  covered by the `spatial` tests and should stay that way.
- **New code under `include/atlas/` / `src/atlas/` carries Doxygen** per the
  coding-style rules; keep this document in sync when the API changes.
