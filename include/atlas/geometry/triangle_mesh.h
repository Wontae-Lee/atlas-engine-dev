#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/container.h>
#include <atlas/geometry/triangle.h>
#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/node.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <limits>
#include <string>

/**
 * @file triangle_mesh.h
 * @brief Arbitrary triangle mesh shape: `TriangleMeshGeometryOperator`
 *        (device-callable queries; a true pointer *view* into the
 *        mesh's vertex/index/BVH buffers, the one exception to
 *        value-embedding among the shape operators — see
 *        `geometry.h`) and its host wrapper `TriangleMesh`
 *        (BVH construction, OBJ loading).
 *
 * @details
 * ### Background — three query families, each with a linear and a
 * ### BVH-accelerated path
 * Every query below has a brute-force `O(triangle_count)` linear
 * implementation and, when a `BVH` (see
 * `spatial/bounding_volume_hierarchy/`) has been built, a
 * `has_bvh()`-gated accelerated path that prunes most of the mesh via
 * the tree's bounding boxes / precomputed moments. `has_bvh()` is
 * forced `false` when compiling *host* code under the CUDA backend
 * (`ATLAS_TASKING_CUDA` without `__CUDA_ARCH__`) because the BVH's
 * pointers are device pointers, not dereferenceable from the host — the
 * operator degrades gracefully to the linear path in that configuration
 * rather than crashing.
 *
 * 1. **Closest point / signed distance** (`closest_point_linear` /
 *    `closest_point_bvh`): a linear scan simply keeps the minimum
 *    `Triangle::closest_point` distance over every
 *    triangle. The BVH path is a standard nearest-neighbor tree
 *    traversal with branch-and-bound pruning: descend into a node only
 *    if `aabb_distance_squared(node.bounds, p) <= best_d2` (a box
 *    farther than the current best cannot contain a closer triangle),
 *    and when both children are viable, visit the nearer one first (so
 *    `best_d2` tightens as early as possible, pruning the farther
 *    subtree more aggressively).
 * 2. **Winding number / inside test** (`winding_number` /
 *    `fast_winding_number_bvh`, feeding `is_inside`/`signed_distance`):
 *    see `node.h`'s Background for the generalized winding number
 *    concept. `solid_angle(p, a, b, c)` is Van Oosterom & Strackee's
 *    (1983) closed-form solid angle subtended by a triangle at a point:
 *    `Omega = 2*atan2(a.(b x c), |a||b||c| + (a.b)|c| + (b.c)|a| +
 *    (c.a)|b|)` (`a, b, c` here are the vertices *relative to* `p`).
 *    Summing this over every triangle and dividing by `4*pi` gives the
 *    exact winding number (`winding_number`). `fast_winding_number_bvh`
 *    accelerates this with the same Barnes-Hut-style multipole
 *    admissibility test used in fast N-body gravity solvers: a node is
 *    "far enough" to approximate as a single point mass
 *    (`approximate_solid_angle`, using the node's precomputed
 *    area-weighted centroid/normal moments, see `node.h`) when its
 *    bounding diagonal is small relative to its distance from `p`
 *    (`size2 <= distance2 * theta^2`, `theta = 0.5` — smaller `theta`
 *    demands the node be relatively farther/smaller before
 *    approximating, trading accuracy for speed); otherwise the
 *    traversal descends into its children, down to exact
 *    per-triangle `solid_angle` sums at the leaves.
 * 3. **Ray tracing** (`trace`): the linear path traces every triangle
 *    and keeps the closest hit. The BVH path is the standard
 *    closest-hit BVH traversal: a node is visited only if its own box
 *    is hit *and* that hit's entry distance doesn't already exceed the
 *    best hit found so far (`hit.enter > best_t` prunes), so the
 *    traversal naturally skips subtrees that cannot possibly contain a
 *    closer intersection.
 *
 * `TriangleContainer4` (the BVH's per-triangle storage) packs a
 * triangle's three vertices plus its precomputed normal (`.d()`) in one
 * value, avoiding a separate normal recomputation per BVH-path query.
 *
 * ### References
 * - A. Van Oosterom and J. Strackee, "The Solid Angle of a Plane
 *   Triangle," IEEE Transactions on Biomedical Engineering BME-30(2),
 *   1983. (the closed-form triangle solid angle `solid_angle()`
 *   implements)
 * - See `node.h`'s References for the generalized winding number and
 *   its BVH-hierarchical fast evaluation.
 */

namespace atlas {

/**
 * @brief Arbitrary triangle mesh device operator; a pointer *view* into
 *        externally-owned vertex/index/BVH buffers (not value-embedded,
 *        unlike every other shape operator — see `geometry.h`).
 *        See this file's top-of-file documentation for the three
 *        query families and their linear/BVH-accelerated paths.
 */
struct TriangleMeshGeometryOperator {

    const Float3* vertices = nullptr;

    const int* indices = nullptr;

    int triangle_count = 0;

    const BVHNode* bvh_nodes = nullptr;

    const int* bvh_indices = nullptr;

    const TriangleContainer4* bvh_tris = nullptr;

    int bvh_root = -1;

private:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    has_bvh() const noexcept {
#if defined(ATLAS_TASKING_CUDA) && !defined(__CUDA_ARCH__)
        return false;
#else
        return bvh_nodes && bvh_indices && bvh_tris && bvh_root >= 0;
#endif
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    closest_point_linear(const Float3& p,
                         Float3* best_point,
                         Float3* best_normal,
                         const float limit) const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return limit;
        }

        float best_d2 = limit;
        Triangle tri {};

        for (int t = 0; t < triangle_count; ++t) {
            const int i0 = indices[3 * t + 0];
            const int i1 = indices[3 * t + 1];
            const int i2 = indices[3 * t + 2];

            const Float3& a = vertices[i0];
            const Float3& b = vertices[i1];
            const Float3& c = vertices[i2];

            tri.a = a;
            tri.b = b;
            tri.c = c;

            const Float3 cp = tri.closest_point(p);
            const float d2  = (cp - p).length_squared();

            if (d2 < best_d2) {
                best_d2 = d2;

                if (best_point) {
                    *best_point = cp;
                }

                if (best_normal) {
                    *best_normal = atlas::normalized_or(
                        atlas::cross(b - a, c - a),
                        Float3(0.0f, 0.0f, 1.0f));
                }
            }
        }

        return best_d2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    closest_point_bvh(const Float3& p,
                      Float3* best_point,
                      Float3* best_normal,
                      const float limit) const noexcept {
        if (!has_bvh()) {
            return limit;
        }

        float best_d2 = limit;

        int stack[64];
        int sp      = 0;
        stack[sp++] = bvh_root;

        while (sp) {
            const int ni      = stack[--sp];
            const BVHNode& nd = bvh_nodes[ni];

            if (atlas::aabb_distance_squared(nd.bounds, p) > best_d2) {
                continue;
            }

            if (nd.is_leaf) {
                Triangle tri_op {};

                ATLAS_UNROLL
                for (int k = 0; k < nd.count; ++k) {
                    const int pid                 = bvh_indices[nd.start + k];
                    const TriangleContainer4& tri = bvh_tris[pid];

                    tri_op.a      = tri.a();
                    tri_op.b      = tri.b();
                    tri_op.c      = tri.c();
                    tri_op.n      = tri.d();
                    tri_op.normal = tri.d();

                    const Float3 cp = tri_op.closest_point(p);
                    const float d2  = (cp - p).length_squared();

                    if (d2 < best_d2) {
                        best_d2 = d2;

                        if (best_point) {
                            *best_point = cp;
                        }

                        if (best_normal) {
                            *best_normal = tri_op.closest_normal(p);
                        }
                    }
                }

                continue;
            }

            const int left  = nd.left;
            const int right = nd.right;

            if (left < 0 && right < 0) {
                continue;
            }

            if (left < 0) {
                if (atlas::aabb_distance_squared(bvh_nodes[right].bounds, p) <= best_d2
                    && sp < 64) {
                    stack[sp++] = right;
                }
                continue;
            }

            if (right < 0) {
                if (atlas::aabb_distance_squared(bvh_nodes[left].bounds, p) <= best_d2
                    && sp < 64) {
                    stack[sp++] = left;
                }
                continue;
            }

            const float left_d2  = atlas::aabb_distance_squared(bvh_nodes[left].bounds, p);
            const float right_d2 = atlas::aabb_distance_squared(bvh_nodes[right].bounds, p);

            const int near_child = (left_d2 <= right_d2) ? left : right;
            const int far_child  = (left_d2 <= right_d2) ? right : left;
            const float near_d2  = (left_d2 <= right_d2) ? left_d2 : right_d2;
            const float far_d2   = (left_d2 <= right_d2) ? right_d2 : left_d2;

            if (far_d2 <= best_d2 && sp < 64) {
                stack[sp++] = far_child;
            }

            if (near_d2 <= best_d2 && sp < 64) {
                stack[sp++] = near_child;
            }
        }

        return best_d2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    approximate_solid_angle(const BVHNode& node,
                            const Float3& p) const noexcept {
        if (!(node.solid_angle_area > 0.0f)) {
            return 0.0f;
        }

        const Float3 center = node.solid_angle_moment / node.solid_angle_area;
        const Float3 r      = center - p;
        const float r2      = r.length_squared();

        if (!(r2 > std::numeric_limits<float>::epsilon())) {
            return 0.0f;
        }

        return node.solid_angle_normal_area.dot(r)
            / (r2 * atlas::sqrt_nonnegative(r2));
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    fast_winding_number_bvh(const Float3& p) const noexcept {
        if (!has_bvh()) {
            return winding_number(p);
        }

        const float theta  = 0.5f;
        const float theta2 = theta * theta;

        float solid_angle_sum = 0.0f;

        int stack[64];
        int sp      = 0;
        stack[sp++] = bvh_root;

        while (sp) {
            const int ni      = stack[--sp];
            const BVHNode& nd = bvh_nodes[ni];

            if (!(nd.solid_angle_area > 0.0f)) {
                continue;
            }

            const Float3 center   = nd.solid_angle_moment / nd.solid_angle_area;
            const float distance2 = (center - p).length_squared();
            const float size2     = nd.bounds.diagonal_length_squared();

            if (!nd.is_leaf
                && distance2 > std::numeric_limits<float>::epsilon()
                && size2 <= distance2 * theta2) {
                solid_angle_sum += approximate_solid_angle(nd, p);
                continue;
            }

            if (nd.is_leaf) {
                ATLAS_UNROLL
                for (int k = 0; k < nd.count; ++k) {
                    const int pid                 = bvh_indices[nd.start + k];
                    const TriangleContainer4& tri = bvh_tris[pid];

                    solid_angle_sum += solid_angle(p, tri.a(), tri.b(), tri.c());
                }

                continue;
            }

            if (nd.left >= 0 && sp < 64) {
                stack[sp++] = nd.left;
            }

            if (nd.right >= 0 && sp < 64) {
                stack[sp++] = nd.right;
            }
        }

        const float four_pi = 4.0f * atlas::pi;
        return solid_angle_sum / four_pi;
    }

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    solid_angle(const Float3& p,
                const Float3& a,
                const Float3& b,
                const Float3& c) const noexcept {
        const Float3 va = a - p;
        const Float3 vb = b - p;
        const Float3 vc = c - p;

        const float la = va.length();
        const float lb = vb.length();
        const float lc = vc.length();

        if (la <= std::numeric_limits<float>::epsilon()
            || lb <= std::numeric_limits<float>::epsilon()
            || lc <= std::numeric_limits<float>::epsilon()) {
            return 0.0f;
        }

        const float numerator = va.dot(atlas::cross(vb, vc));

        const float denominator = la * lb * lc
            + va.dot(vb) * lc
            + vb.dot(vc) * la
            + vc.dot(va) * lb;

        return 2.0f * std::atan2(numerator, denominator);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    winding_number(const Float3& p) const noexcept {
        float solid_angle_sum = 0.0f;

        for (int t = 0; t < triangle_count; ++t) {

            const int i0 = indices[3 * t + 0];
            const int i1 = indices[3 * t + 1];
            const int i2 = indices[3 * t + 2];

            const Float3& a = vertices[i0];
            const Float3& b = vertices[i1];
            const Float3& c = vertices[i2];

            solid_angle_sum += solid_angle(p, a, b, c);
        }

        const float four_pi = 4.0f * atlas::pi;

        return solid_angle_sum / four_pi;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return p;
        }

        Float3 best_cp = p;

        if (has_bvh()) {
            closest_point_bvh(p, &best_cp, nullptr, std::numeric_limits<float>::max());
        } else {
            closest_point_linear(p, &best_cp, nullptr, std::numeric_limits<float>::max());
        }

        return best_cp;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3& p) const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return Float3(0.0f, 0.0f, 1.0f);
        }

        Float3 best_n(0.0f, 0.0f, 1.0f);

        if (has_bvh()) {
            closest_point_bvh(p, nullptr, &best_n, std::numeric_limits<float>::max());
        } else {
            closest_point_linear(p, nullptr, &best_n, std::numeric_limits<float>::max());
        }

        return best_n;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return std::numeric_limits<float>::infinity();
        }

        const float best_d2 = has_bvh()
            ? closest_point_bvh(p, nullptr, nullptr, std::numeric_limits<float>::max())
            : closest_point_linear(p, nullptr, nullptr, std::numeric_limits<float>::max());

        const float dist = atlas::sqrt_nonnegative(best_d2);

        if (dist <= std::numeric_limits<float>::epsilon()) {
            return 0.0f;
        }

        const float winding = has_bvh()
            ? fast_winding_number_bvh(p)
            : winding_number(p);
        const bool inside   = std::abs(winding) > 0.5f;

        return inside ? -dist : dist;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return false;
        }

        const float winding = has_bvh()
            ? fast_winding_number_bvh(p)
            : winding_number(p);
        const bool inside   = std::abs(winding) > 0.5f;

        if (inside && tolerance >= 0.0f) {
            return true;
        }

        if (!inside && tolerance < 0.0f) {
            return false;
        }

        const float tolerance2 = tolerance * tolerance;
        const float limit      = tolerance2
            + ((tolerance2 > 1.0f) ? tolerance2 : 1.0f) * std::numeric_limits<float>::epsilon();
        const float best_d2 = has_bvh()
            ? closest_point_bvh(p, nullptr, nullptr, limit)
            : closest_point_linear(p, nullptr, nullptr, limit);

        return inside ? best_d2 >= tolerance2
                      : best_d2 <= tolerance2;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!vertices || !indices || triangle_count <= 0 || tolerance < 0.0f) {
            return false;
        }

        const float tolerance2 = tolerance * tolerance;
        const float limit      = tolerance2
            + ((tolerance2 > 1.0f) ? tolerance2 : 1.0f) * std::numeric_limits<float>::epsilon();
        const float best_d2 = has_bvh()
            ? closest_point_bvh(p, nullptr, nullptr, limit)
            : closest_point_linear(p, nullptr, nullptr, limit);

        return best_d2 <= tolerance2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return Float3(0.0f, 0.0f, 0.0f);
        }

        Float3 sum(0.0f, 0.0f, 0.0f);

        for (int t = 0; t < triangle_count; ++t) {

            const int i0 = indices[3 * t + 0];
            const int i1 = indices[3 * t + 1];
            const int i2 = indices[3 * t + 2];

            const Float3& a = vertices[i0];
            const Float3& b = vertices[i1];
            const Float3& c = vertices[i2];

            sum += (a + b + c) * (1.0f / 3.0f);
        }

        return sum * (1.0f / static_cast<float>(triangle_count));
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return AABB();
        }

        const int i0 = indices[0];
        Float3 mn    = vertices[i0];
        Float3 mx    = vertices[i0];

        for (int t = 0; t < triangle_count; ++t) {

            const int j0 = indices[3 * t + 0];
            const int j1 = indices[3 * t + 1];
            const int j2 = indices[3 * t + 2];

            mn = atlas::cmin(mn, vertices[j0]);
            mn = atlas::cmin(mn, vertices[j1]);
            mn = atlas::cmin(mn, vertices[j2]);

            mx = atlas::cmax(mx, vertices[j0]);
            mx = atlas::cmax(mx, vertices[j1]);
            mx = atlas::cmax(mx, vertices[j2]);
        }

        return AABB(mn, mx);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        if (!vertices || !indices) {
            return false;
        }

        if (triangle_count <= 0) {
            return false;
        }

        for (int t = 0; t < triangle_count; ++t) {

            const int j0 = indices[3 * t + 0];
            const int j1 = indices[3 * t + 1];
            const int j2 = indices[3 * t + 2];

            const Float3 ab = vertices[j1] - vertices[j0];
            const Float3 ac = vertices[j2] - vertices[j0];
            const Float3 n  = atlas::cross(ab, ac);

            if (!(n.length_squared() > atlas::eps)) {
                return false;
            }
        }

        return true;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface out {};

#if defined(ATLAS_TASKING_CUDA) && !defined(__CUDA_ARCH__)
        constexpr bool can_use_bvh = false;
#else
        constexpr bool can_use_bvh = true;
#endif

        if (!can_use_bvh) {
            if (!vertices || !indices || triangle_count <= 0) {
                return out;
            }

            float best_t = std::numeric_limits<float>::max();
            Triangle tri_op {};

            for (int t = 0; t < triangle_count; ++t) {
                const int i0 = indices[3 * t + 0];
                const int i1 = indices[3 * t + 1];
                const int i2 = indices[3 * t + 2];

                const Float3& a = vertices[i0];
                const Float3& b = vertices[i1];
                const Float3& c = vertices[i2];

                tri_op.a      = a;
                tri_op.b      = b;
                tri_op.c      = c;
                tri_op.n      = atlas::cross(b - a, c - a);
                tri_op.normal = tri_op.n;

                const HitSurface h = tri_op.trace(r);
                if (h.is_intersecting && h.distance < best_t) {
                    best_t = h.distance;
                    out    = h;
                }
            }

            return out;
        }

        if (!bvh_nodes || !bvh_indices || !bvh_tris || bvh_root < 0) {
            return out;
        }

        float best_t = std::numeric_limits<float>::max();
        Float3 best_p;
        Float3 best_n;
        bool found = false;

        int stack[64];
        int sp      = 0;
        stack[sp++] = bvh_root;

        while (sp) {

            const int ni      = stack[--sp];
            const BVHNode& nd = bvh_nodes[ni];

            const auto hit = nd.bounds.trace(r);
            if (!hit.is_intersecting || hit.enter > best_t) {
                continue;
            }

            if (nd.is_leaf) {
                Triangle tri_op {};

                ATLAS_UNROLL
                for (int k = 0; k < nd.count; ++k) {

                    const int pid                 = bvh_indices[nd.start + k];
                    const TriangleContainer4& tri = bvh_tris[pid];

                    tri_op.a      = tri.a();
                    tri_op.b      = tri.b();
                    tri_op.c      = tri.c();
                    tri_op.n      = tri.d();
                    tri_op.normal = tri.d();

                    const HitSurface h = tri_op.trace(r);

                    if (h.is_intersecting && h.distance < best_t) {
                        best_t = h.distance;
                        best_p = h.point;
                        best_n = h.normal;
                        found  = true;
                    }
                }
            } else {

                if (sp < 63) {
                    stack[sp++] = nd.left;
                } else {
                    stack[63] = nd.left;
                }

                if (sp < 63) {
                    stack[sp++] = nd.right;
                } else {
                    stack[63] = nd.right;
                }
            }
        }

        if (found) {

            out.is_intersecting = true;
            out.distance        = best_t;
            out.point           = best_p;
            out.normal          = best_n;
        }

        return out;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }
};

/**
 * @brief Host-side triangle mesh `Geometry`: owns the triangle data,
 *        lazily builds a `SAHBVH` acceleration structure
 *        (`ensure_bvh`/`build_bvh`), and maintains a flat
 *        vertex/index "query cache" (`ensure_query_cache`/
 *        `rebuild_query_cache`) so the device-callable
 *        `TriangleMeshGeometryOperator` this produces
 *        (`make_device_geometry_view`) can view the mesh through plain
 *        pointers. `triangles` stores each triangle independently (its
 *        own three vertices, no shared-vertex adjacency) — the query
 *        cache's `indices` are therefore just `0..3n-1` identity
 *        indices, present only to match the operator's
 *        vertices+indices interface.
 */
struct Geometry;

class TriangleMesh final {
public:
    class Builder;

public:
    HostBuffer<TriangleContainer4> triangles;

    TriangleMesh() noexcept = default;

    ATLAS_HOST explicit TriangleMesh(const HostBuffer<TriangleContainer4>& triangles_) noexcept;

    ATLAS_HOST explicit TriangleMesh(HostBuffer<TriangleContainer4>&& triangles_) noexcept;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_HOST
    TriangleMesh(const TriangleMesh& other);

    ATLAS_HOST
    TriangleMesh(TriangleMesh&& other) noexcept;

    ATLAS_HOST TriangleMesh&
    operator=(const TriangleMesh& other);

    ATLAS_HOST TriangleMesh&
    operator=(TriangleMesh&& other) noexcept;

    ~TriangleMesh() = default;

    ATLAS_HOST void
    set_triangles(const HostBuffer<TriangleContainer4>& triangles_);

    ATLAS_NODISCARD ATLAS_HOST Geometry
    make_device_geometry_view() const;

    /** @brief Loads triangle data from a Wavefront OBJ file at
     *  `filename`, replacing `triangles`; `false` on failure. */
    ATLAS_NODISCARD ATLAS_HOST bool
    load_from_obj(const std::string& filename, bool verbose = false);

    ATLAS_HOST Float3
    closest_point(const Float3& p) const noexcept;

    ATLAS_HOST Float3
    closest_normal(const Float3& p) const noexcept;

    ATLAS_HOST float
    signed_distance(const Float3& p) const noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    is_inside(const Float3& p, float tolerance) const noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    is_on_surface(const Float3& p, float tolerance) const noexcept;

    ATLAS_HOST Float3
    centroid() const noexcept;

    ATLAS_HOST AABB
    bound() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    is_valid() const noexcept;

private:
    friend class Builder;

private:
    BVHHostPtr _bvh = nullptr;

    mutable HostBuffer<Float3> _query_vertices;

    mutable HostBuffer<int> _query_indices;

    bool bvh_built = false;

    mutable bool query_cache_built = false;

    mutable TriangleMeshGeometryOperator _operator {};

    /** @brief Lazily allocates `_bvh` as a `SAHBVH` if not already
     *  present (the default/only BVH strategy `TriangleMesh` uses). */
    ATLAS_HOST void
    ensure_bvh() noexcept;

    /** @brief Builds (or rebuilds) `_bvh` over the current `triangles`;
     *  `bvh_built` gates whether `make_device_geometry_view` exposes
     *  BVH pointers at all. */
    ATLAS_HOST void
    build_bvh();

    /** @brief Rebuilds the flat vertex/index query cache only if it is
     *  not already up to date (`query_cache_built`). */
    ATLAS_HOST void
    ensure_query_cache() const;

    /** @brief Unconditionally regenerates `_query_vertices`/
     *  `_query_indices` from `triangles` (see this class's top-of-class
     *  documentation for why the indices are just `0..3n-1`). */
    ATLAS_HOST void
    rebuild_query_cache() const;

    /** @brief Refreshes `_operator`'s pointers/counts from the current
     *  query cache and (if built) BVH. */
    ATLAS_HOST void
    update_operator() const;
};

/** @brief Fluent builder for `TriangleMesh`; set triangles directly
 *  (`with_triangles`) or load them from an OBJ file
 *  (`load_from_obj`). */
class TriangleMesh::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST TriangleMesh
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<TriangleMesh>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_triangles(const HostBuffer<TriangleContainer4>& ts);

    ATLAS_HOST Builder&
    with_triangles(HostBuffer<TriangleContainer4>&& ts);

    ATLAS_HOST Builder&
    load_from_obj(const std::string& filename, bool verbose = false);

private:
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<TriangleContainer4> _triangles;
};

using TriangleMeshF = TriangleMesh;

using TriangleMeshHostPtr = atlas::host_shared_ptr<TriangleMesh>;

using TriangleMeshDevicePtr = atlas::device_shared_ptr<TriangleMesh>;

}
