#pragma once

/**
 * @file triangle_mesh.h
 * @brief Triangle-soup geometry leaf and its host-side owner.
 *
 * This header declares two cooperating types:
 *
 *  - @ref atlas::TriangleMeshView — a trivially-copyable struct of raw pointers
 *    (vertices, indices, and an optional BVH) that answers all geometric
 *    queries (closest point/normal, signed distance, inside/surface tests,
 *    ray trace, winding number). Because it is trivially copyable and every
 *    query is `ATLAS_ALL_DEVICE`, it can be captured by value inside a device
 *    lambda and evaluated on the GPU. It is the `triangle_mesh` case of the
 *    @ref atlas::Geometry tagged union.
 *  - @ref atlas::TriangleMesh — the host-only owner that holds the triangle
 *    buffer, lazily builds a SAH BVH and a flat query-vertex cache, keeps a
 *    @ref atlas::TriangleMeshView pointing into them, and hands that view to
 *    the device via @ref atlas::TriangleMesh::make_device_geometry_view.
 *
 * @note The view stores no ownership: its lifetime is bound to the owning
 *       @ref atlas::TriangleMesh (or the BVH it was built from). A view whose
 *       backing buffers were freed or reallocated dangles.
 * @see atlas::Geometry, atlas::BVH, atlas::SAHBVH
 */

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

namespace atlas {

/**
 * @brief Trivially-copyable, device-capturable view over a triangle mesh.
 *
 * Every field is a raw pointer or plain scalar gathered once on the host so a
 * single copy of this struct can be captured by value and dereferenced inside a
 * device lambda. All query methods are `ATLAS_ALL_DEVICE` and const; the struct
 * owns nothing and must not outlive the buffers it points into.
 *
 * Two independent geometry representations are held at once:
 *
 *  - A flat triangle soup: @ref vertices indexed by @ref indices, three indices
 *    per triangle. This is always populated by the owning mesh and is the
 *    fallback used when no acceleration structure is available.
 *  - An optional bounding-volume hierarchy (@ref bvh_nodes, @ref bvh_indices,
 *    @ref bvh_tris, @ref bvh_root). When present it accelerates closest-point,
 *    ray-trace, and fast-winding-number queries.
 *
 * @note This alias is also reused as @ref atlas::BvhView: a BVH's `view()`
 *       returns a `TriangleMeshView` carrying only its acceleration fields.
 * @warning In a CUDA build the BVH pointers refer to device memory, so
 *          @ref has_bvh deliberately reports "no BVH" on the host and the
 *          host-side query falls back to the flat triangle soup. See
 *          @ref has_bvh.
 */
struct TriangleMeshView {

    /// Flat vertex positions; indexed by @ref indices. Null when the mesh is empty.
    const Float3* vertices = nullptr;

    /// Vertex indices, three consecutive entries per triangle. Null when empty.
    const int* indices = nullptr;

    /// Number of triangles, i.e. `indices` has `3 * triangle_count` entries.
    int triangle_count = 0;

    /// BVH node array; null when no BVH is attached. Device memory in CUDA builds.
    const BVHNode* bvh_nodes = nullptr;

    /// Permutation mapping a leaf's `[start, start+count)` slots to triangle ids.
    const int* bvh_indices = nullptr;

    /// Triangles in BVH storage order; each carries its precomputed normal in `d()`.
    const TriangleContainer4* bvh_tris = nullptr;

    /// Index of the BVH root node, or -1 when no BVH is attached.
    int bvh_root = -1;

private:
    /**
     * @brief Whether the BVH acceleration structure is usable on the caller side.
     *
     * @return `true` when all BVH pointers are non-null and the root is valid.
     *
     * @note In a CUDA build (`ATLAS_BACKEND_CUDA`) evaluated on the host
     *       (`!__CUDA_ARCH__`) this always returns `false`, because the BVH
     *       buffers live in device memory and cannot be dereferenced from the
     *       host. Host-side queries then use the flat triangle soup instead,
     *       while the same call on the device sees the real BVH. In a CPU
     *       tasking build the BVH buffers are host-accessible and this returns
     *       the real availability on both sides.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    has_bvh() const noexcept {
#if defined(ATLAS_BACKEND_CUDA) && !defined(__CUDA_ARCH__)
        return false;
#else
        return bvh_nodes && bvh_indices && bvh_tris && bvh_root >= 0;
#endif
    }

    /**
     * @brief Brute-force closest-point scan over every triangle.
     *
     * Walks all `triangle_count` triangles of the flat soup, tracking the
     * smallest squared distance from @p p to any triangle. Used whenever the
     * BVH is unavailable (empty mesh, CPU build without a BVH, or the host side
     * of a CUDA build).
     *
     * @param p          Query point.
     * @param best_point If non-null, receives the closest surface point found.
     * @param best_normal If non-null, receives the geometric normal of the
     *                    closest triangle (`cross(b-a, c-a)` normalized, falling
     *                    back to +Z for a degenerate triangle).
     * @param limit      Initial squared-distance bound; only triangles closer
     *                   than this are considered. Pass `FLT_MAX` for an
     *                   unbounded search, or `tolerance^2` to early-reject.
     * @return The smallest squared distance found, or @p limit if the mesh is
     *         empty or nothing beats the bound. Never returns a raw distance.
     */
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

    /**
     * @brief BVH-accelerated closest-point query.
     *
     * Iterative depth-first traversal using a fixed 64-entry stack (no
     * recursion, so it is valid on the device). A node is skipped when its
     * AABB is already farther than the current best; leaves test each contained
     * triangle. Children are pushed farthest-first so the nearer child is
     * popped and expanded first, tightening the bound sooner. The normal
     * returned comes from the triangle's stored/interpolated
     * `Triangle::closest_normal`, unlike the linear path's geometric normal.
     *
     * @param p          Query point.
     * @param best_point If non-null, receives the closest surface point found.
     * @param best_normal If non-null, receives the closest triangle's normal.
     * @param limit      Initial squared-distance bound (see
     *                   @ref closest_point_linear).
     * @return The smallest squared distance found, or @p limit when no BVH is
     *         usable or nothing beats the bound.
     * @warning The stack is capped at 64 entries; pushes are silently dropped
     *          when full (`sp < 64` guards). For a well-balanced SAH BVH 64
     *          levels is far beyond any realistic depth, so this is a safety
     *          bound rather than an expected path.
     */
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

            // Push far first, near last: the LIFO stack then pops the nearer
            // child first, tightening best_d2 before the far subtree is visited.
            if (far_d2 <= best_d2 && sp < 64) {
                stack[sp++] = far_child;
            }

            if (near_d2 <= best_d2 && sp < 64) {
                stack[sp++] = near_child;
            }
        }

        return best_d2;
    }

    /**
     * @brief Dipole approximation of a node's solid angle (Barill et al.).
     *
     * Treats the triangles beneath @p node as a single oriented point source at
     * the area-weighted centroid `solid_angle_moment / solid_angle_area`, and
     * evaluates the far-field solid angle
     * `(normal_area · r) / |r|^3` where `r = center - p`. Used by
     * @ref fast_winding_number_bvh for nodes far enough that this cheap
     * approximation is accurate.
     *
     * @param node BVH node carrying the precomputed solid-angle moments.
     * @param p    Query point.
     * @return The approximate signed solid angle contribution, or 0 when the
     *         node has no area or @p p sits (numerically) at the center.
     */
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

    /**
     * @brief Barnes-Hut style fast winding number using the BVH.
     *
     * Accumulates the generalized winding number of @p p against the mesh by
     * traversing the BVH: a well-separated interior node (its cluster size
     * small relative to its distance, `size2 <= distance2 * theta2`) is
     * summarized by @ref approximate_solid_angle instead of being descended,
     * while leaves sum the exact triangle solid angles. The total is divided by
     * `4*pi`, so ~1 inside a closed surface and ~0 outside.
     *
     * @param p Query point.
     * @return The generalized winding number; falls back to the exact
     *         @ref winding_number when no BVH is usable.
     * @note `theta = 0.5` is the Barnes-Hut opening criterion: larger accepts
     *       coarser approximations (faster, less accurate).
     */
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
    /**
     * @brief Signed solid angle subtended by a single triangle at a point.
     *
     * Uses the numerically stable van Oosterom-Strackee formula
     * `2*atan2(va · (vb × vc), |va||vb||vc| + (va·vb)|vc| + ...)`, where
     * `va,vb,vc` are the vertices relative to @p p. The sign follows the
     * triangle's winding, so summing these over a closed, consistently oriented
     * mesh yields ±4*pi inside and 0 outside.
     *
     * @param p Query point (apex of the solid angle).
     * @param a First triangle vertex.
     * @param b Second triangle vertex.
     * @param c Third triangle vertex.
     * @return The signed solid angle in steradians, or 0 when @p p coincides
     *         with a vertex (degenerate, guarded by an epsilon on edge lengths).
     */
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

    /**
     * @brief Exact generalized winding number by summing every triangle.
     *
     * Brute-force `O(triangle_count)` accumulation of @ref solid_angle over the
     * flat soup, divided by `4*pi`. Approaches ±1 for points inside a closed
     * mesh and 0 outside; a robust inside test even for non-watertight meshes.
     *
     * @param p Query point.
     * @return The generalized winding number.
     * @warning Performs no null/empty guard on @ref vertices / @ref indices;
     *          callers (@ref signed_distance, @ref is_inside) check emptiness
     *          first, and an empty mesh simply yields the initial 0.
     */
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

    /**
     * @brief Closest point on the mesh surface to @p p.
     *
     * Dispatches to @ref closest_point_bvh when a BVH is usable, else to
     * @ref closest_point_linear.
     *
     * @param p Query point.
     * @return The nearest surface point, or @p p unchanged when the mesh is
     *         empty (degenerate no-op so callers get a sane value).
     */
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

    /**
     * @brief Surface normal of the triangle closest to @p p.
     *
     * @param p Query point.
     * @return The closest triangle's normal, or +Z `(0,0,1)` when the mesh is
     *         empty. Note the BVH and linear paths derive the normal
     *         differently (stored/interpolated vs. geometric `cross(b-a,c-a)`).
     */
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

    /**
     * @brief Signed distance from @p p to the mesh surface.
     *
     * Computes the unsigned distance via the closest-point query, then flips
     * its sign using the winding number: negative inside a (closed) mesh,
     * positive outside.
     *
     * @param p Query point.
     * @return Signed distance; `+inf` for an empty mesh, `0` when @p p lies on
     *         the surface (within epsilon). Sign uses the `|winding| > 0.5`
     *         inside test, which is only meaningful for closed meshes.
     */
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

    /**
     * @brief Inside test with a signed surface tolerance.
     *
     * Combines the winding-number inside test with a distance-to-surface check
     * so the boundary can be widened or narrowed:
     *
     *  - `tolerance == 0`: plain inside test (`|winding| > 0.5`).
     *  - `tolerance > 0`: dilates the solid — an exterior point still counts as
     *    inside if it is within @p tolerance of the surface.
     *  - `tolerance < 0`: erodes the solid — an interior point counts as inside
     *    only if it is at least `|tolerance|` away from the surface.
     *
     * @param p         Query point.
     * @param tolerance Signed shell width (default 0). See above.
     * @return `true` when @p p is inside under the given tolerance; `false` for
     *         an empty mesh. The distance test uses `tolerance^2` with a tiny
     *         relative epsilon added to the search @c limit so borderline points
     *         are not rejected by rounding.
     */
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

    /**
     * @brief Whether @p p lies within @p tolerance of the mesh surface.
     *
     * A pure proximity test: returns true when the closest surface point is no
     * farther than @p tolerance, regardless of inside/outside. Uses the same
     * epsilon-padded @c limit as @ref is_inside so the search can early-reject
     * distant points.
     *
     * @param p         Query point.
     * @param tolerance Non-negative shell half-width (default 0).
     * @return `true` when within tolerance of the surface; `false` for an empty
     *         mesh or a negative @p tolerance (rejected as invalid).
     */
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

    /**
     * @brief Unweighted mean of the per-triangle barycenters.
     *
     * Averages `(a+b+c)/3` over every triangle. This is a triangle-count
     * average, not an area-weighted or volume centroid, so denser tessellation
     * pulls the result toward the finer region.
     *
     * @return The averaged center, or the origin for an empty mesh.
     */
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

    /**
     * @brief Axis-aligned bounding box enclosing every referenced vertex.
     *
     * Uses the BVH root's bound when the hierarchy is accessible to the caller.
     * Otherwise seeds min/max from the first indexed vertex, then folds in all
     * three vertices of every triangle with @c cmin / @c cmax.
     *
     * @return The enclosing AABB, or a default (empty) AABB for an empty mesh.
     * @note The CUDA device path reads the device BVH rather than the owner's
     *       host-only flat query cache, including when a collider advances.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        if (!vertices || !indices || triangle_count <= 0) {
            return AABB();
        }

        if (has_bvh()) {
            return bvh_nodes[bvh_root].bounds;
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

    /**
     * @brief Whether the mesh is present and free of degenerate triangles.
     *
     * Requires non-null buffers, a positive triangle count, and every triangle
     * to have a non-zero cross product (`|cross(ab,ac)|^2 > eps`), i.e. no
     * zero-area/collinear triangles.
     *
     * @return `true` if all checks pass; `false` on the first failure.
     */
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

    /**
     * @brief Ray-cast the mesh and return the nearest surface hit.
     *
     * On the device (or a CPU build) traverses the BVH, culling nodes whose
     * AABB entry distance already exceeds the best hit and testing leaf
     * triangles with @c Triangle::trace. On the host side of a CUDA build the
     * BVH pointers are device memory, so @c can_use_bvh is `false` and it falls
     * back to a brute-force scan over the flat soup (recomputing each triangle's
     * geometric normal on the fly).
     *
     * @param r Ray to cast (origin + normalized direction).
     * @return The closest @ref HitSurface; its `is_intersecting` is `false` when
     *         nothing was hit or the mesh/BVH is unavailable.
     * @warning The traversal stack is capped at 64 entries. Unlike the
     *          closest-point traversal, on overflow it overwrites `stack[63]`
     *          rather than dropping the push, so on a pathologically deep tree
     *          some nodes can be lost and a hit missed. This is a safety bound;
     *          a balanced SAH BVH never approaches this depth.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface out {};

        // Host side of a CUDA build cannot dereference the device BVH buffers;
        // fall back to the flat-soup scan below. Device / CPU builds use the BVH.
#if defined(ATLAS_BACKEND_CUDA) && !defined(__CUDA_ARCH__)
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
};

/// Forward declaration of the geometry tagged union this mesh feeds a view into.
struct Geometry;

/**
 * @brief Host-only owner of a triangle mesh and its query acceleration.
 *
 * Owns the triangle buffer plus two lazily-built, cached side structures:
 *
 *  - a SAH BVH (@ref _bvh) for accelerated closest-point / trace / winding
 *    queries, and
 *  - a flat query cache (@ref _query_vertices, @ref _query_indices) — the
 *    triangles expanded into a plain vertex+index soup so a
 *    @ref TriangleMeshView can point at contiguous host memory.
 *
 * A cached @ref _view is kept in sync with those buffers and handed to the
 * device through @ref make_device_geometry_view. The host-side query methods
 * (@ref closest_point, @ref signed_distance, ...) simply forward to that view.
 *
 * @note Copy and move are user-provided because they must rebuild @ref _view to
 *       point into *this* object's buffers rather than the source's.
 * @see TriangleMeshView, Geometry, Builder
 */
class TriangleMesh final {
public:
    /// Fluent construction helper; see @ref TriangleMesh::Builder.
    class Builder;

public:
    /// The mesh triangles, each storing its three corners and a normal in `d()`.
    HostBuffer<TriangleContainer4> triangles;

    /// Constructs an empty mesh; no BVH or cache is built until triangles arrive.
    TriangleMesh() noexcept = default;

    /**
     * @brief Constructs from a copy of a triangle buffer, building BVH and cache.
     * @param triangles_ Triangles to copy in.
     */
    ATLAS_HOST explicit TriangleMesh(const HostBuffer<TriangleContainer4>& triangles_) noexcept;

    /**
     * @brief Constructs by moving a triangle buffer, building BVH and cache.
     * @param triangles_ Triangles to take ownership of.
     */
    ATLAS_HOST explicit TriangleMesh(HostBuffer<TriangleContainer4>&& triangles_) noexcept;

    /**
     * @brief Returns a fresh @ref Builder.
     * @return A default-constructed builder.
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Copy constructor; duplicates buffers and shares the BVH pointer,
     *        then rebuilds the view to point into this instance.
     * @param other Mesh to copy.
     */
    ATLAS_HOST
    TriangleMesh(const TriangleMesh& other);

    /**
     * @brief Move constructor; steals buffers and BVH, rebuilds both views, and
     *        resets @p other to an empty, view-cleared state.
     * @param other Mesh to move from.
     */
    ATLAS_HOST
    TriangleMesh(TriangleMesh&& other) noexcept;

    /**
     * @brief Copy assignment; self-assignment safe. Rebuilds the view afterwards.
     * @param other Mesh to copy.
     * @return `*this`.
     */
    ATLAS_HOST TriangleMesh&
    operator=(const TriangleMesh& other);

    /**
     * @brief Move assignment; self-assignment safe. Rebuilds both views and
     *        clears @p other.
     * @param other Mesh to move from.
     * @return `*this`.
     */
    ATLAS_HOST TriangleMesh&
    operator=(TriangleMesh&& other) noexcept;

    /// Defaulted; all owned members clean up themselves.
    ~TriangleMesh() = default;

    /**
     * @brief Replaces the triangles and rebuilds the BVH and query cache.
     * @param triangles_ New triangles to copy in.
     */
    ATLAS_HOST void
    set_triangles(const HostBuffer<TriangleContainer4>& triangles_);

    /**
     * @brief Produces a @ref Geometry wrapping this mesh's device-capturable view.
     *
     * Ensures the query cache is current, then wraps the cached
     * @ref TriangleMeshView in a @ref Geometry (the `triangle_mesh` case).
     *
     * @return A `Geometry` value holding a view whose pointers alias this mesh's
     *         buffers; it stays valid only while this mesh (and its buffers)
     *         live and are not rebuilt.
     */
    ATLAS_NODISCARD ATLAS_HOST Geometry
    make_device_geometry_view() const;

    /**
     * @brief Loads triangles from a Wavefront OBJ file (triangulated).
     *
     * Clears existing triangles, parses @p filename via tiny_obj_loader, keeps
     * only faces with valid, in-range vertex indices, computes a per-face normal
     * (falling back to +Z for a degenerate face), then rebuilds BVH and cache.
     *
     * @param filename Path to the OBJ file.
     * @param verbose  Currently ignored (reserved; suppressed via `(void)`).
     * @return `true` on success; `false` if parsing fails, the file has no
     *         vertices, or no valid triangle survived.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    load_from_obj(const std::string& filename, bool verbose = false);

    /**
     * @brief Host-side closest surface point; forwards to the cached view.
     * @param p Query point.
     * @return Nearest surface point (see @ref TriangleMeshView::closest_point).
     */
    ATLAS_HOST Float3
    closest_point(const Float3& p) const noexcept;

    /**
     * @brief Host-side closest surface normal; forwards to the cached view.
     * @param p Query point.
     * @return Nearest triangle's normal (see @ref TriangleMeshView::closest_normal).
     */
    ATLAS_HOST Float3
    closest_normal(const Float3& p) const noexcept;

    /**
     * @brief Host-side signed distance; forwards to the cached view.
     * @param p Query point.
     * @return Signed distance (see @ref TriangleMeshView::signed_distance).
     */
    ATLAS_HOST float
    signed_distance(const Float3& p) const noexcept;

    /**
     * @brief Host-side inside test with signed tolerance; forwards to the view.
     * @param p         Query point.
     * @param tolerance Signed shell width (see @ref TriangleMeshView::is_inside).
     * @return `true` when inside under the tolerance.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    is_inside(const Float3& p, float tolerance) const noexcept;

    /**
     * @brief Host-side surface-proximity test; forwards to the view.
     * @param p         Query point.
     * @param tolerance Non-negative shell half-width.
     * @return `true` when within tolerance of the surface.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    is_on_surface(const Float3& p, float tolerance) const noexcept;

    /**
     * @brief Host-side centroid; forwards to the cached view.
     * @return Triangle-count average of per-triangle barycenters.
     */
    ATLAS_HOST Float3
    centroid() const noexcept;

    /**
     * @brief Host-side bounding box; forwards to the cached view.
     * @return AABB enclosing all vertices.
     */
    ATLAS_HOST AABB
    bound() const noexcept;

    /**
     * @brief Host-side validity check; forwards to the cached view.
     * @return `true` when the mesh is non-empty and free of degenerate triangles.
     */
    ATLAS_NODISCARD ATLAS_HOST bool
    is_valid() const noexcept;

private:
    /// Grants the nested builder access to the private cache-priming helpers.
    friend class Builder;

private:
    /// Shared-owned acceleration structure; lazily allocated by @ref ensure_bvh.
    BVHHostPtr _bvh = nullptr;

    /// Flat query vertices (3 per triangle); backs the view when the BVH is off.
    mutable HostBuffer<Float3> _query_vertices;

    /// Flat query indices (0,1,2,...) paired with @ref _query_vertices.
    mutable HostBuffer<int> _query_indices;

    /// Whether @ref _bvh currently holds a built hierarchy for @ref triangles.
    bool bvh_built = false;

    /// Whether the flat query cache matches @ref triangles (mutable: lazy build).
    mutable bool query_cache_built = false;

    /// Device-capturable view kept in sync with the buffers above.
    mutable TriangleMeshView _view {};

    /**
     * @brief Lazily allocates the BVH object (a @ref SAHBVH) if none exists.
     * @note Only creates the object; @ref build_bvh populates it.
     */
    ATLAS_HOST void
    ensure_bvh() noexcept;

    /**
     * @brief Builds the BVH over the current triangles.
     *
     * Sets @ref bvh_built; leaves it `false` (and does nothing) when there is no
     * BVH object or the triangle buffer is empty.
     */
    ATLAS_HOST void
    build_bvh();

    /**
     * @brief Rebuilds the flat query cache only if it is stale.
     * @note Const because the cache and its flag are `mutable`.
     */
    ATLAS_HOST void
    ensure_query_cache() const;

    /**
     * @brief Expands @ref triangles into the flat vertex/index query cache.
     *
     * Writes 3 vertices and sequential indices per triangle (vertices are not
     * deduplicated), marks the cache built, and refreshes @ref _view.
     */
    ATLAS_HOST void
    rebuild_query_cache() const;

    /**
     * @brief Repoints @ref _view at the current buffers and BVH.
     *
     * Sets the soup pointers (null when empty) and either the live BVH view
     * fields or their empty sentinels depending on @ref bvh_built.
     */
    ATLAS_HOST void
    update_view() const;
};

/**
 * @brief Fluent builder for @ref TriangleMesh.
 *
 * Accumulates a triangle buffer via `with_triangles` / `load_from_obj`, then
 * `validate()`s and constructs the mesh (which builds its BVH and cache).
 */
class TriangleMesh::Builder final {
public:
    /// Constructs an empty builder with no triangles staged.
    Builder() = default;

    /**
     * @brief Validates and builds the mesh.
     * @return A fully constructed @ref TriangleMesh (BVH and cache built).
     * @throws std::runtime_error when no triangles were staged.
     */
    ATLAS_NODISCARD ATLAS_HOST TriangleMesh
    build() const;

    /**
     * @brief Builds the mesh and wraps it in a host shared pointer.
     * @return `host_shared_ptr` owning the built mesh.
     * @throws std::runtime_error when no triangles were staged.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<TriangleMesh>
    make_host_shared() const;

    /**
     * @brief Stages a copy of a triangle buffer.
     * @param ts Triangles to copy.
     * @return `*this` for chaining.
     */
    ATLAS_HOST Builder&
    with_triangles(const HostBuffer<TriangleContainer4>& ts);

    /**
     * @brief Stages a triangle buffer by move.
     * @param ts Triangles to take.
     * @return `*this` for chaining.
     */
    ATLAS_HOST Builder&
    with_triangles(HostBuffer<TriangleContainer4>&& ts);

    /**
     * @brief Stages triangles loaded from an OBJ file.
     * @param filename Path to the OBJ file.
     * @param verbose  Currently ignored.
     * @return `*this` for chaining.
     * @throws std::runtime_error when the file fails to load.
     */
    ATLAS_HOST Builder&
    load_from_obj(const std::string& filename, bool verbose = false);

private:
    /**
     * @brief Throws when the staged triangle buffer is empty.
     * @throws std::runtime_error on an empty buffer.
     */
    ATLAS_HOST void
    validate() const;

private:
    /// Triangles staged for the next @ref build.
    HostBuffer<TriangleContainer4> _triangles;
};

/// Convenience alias: the float-precision mesh (the only precision provided).
using TriangleMeshF = TriangleMesh;

/// Host shared-ownership pointer to a @ref TriangleMesh.
using TriangleMeshHostPtr = atlas::host_shared_ptr<TriangleMesh>;

/// Device shared-ownership pointer to a @ref TriangleMesh.
using TriangleMeshDevicePtr = atlas::device_shared_ptr<TriangleMesh>;

}
