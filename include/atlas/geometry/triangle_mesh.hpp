#pragma once

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>
#include <atlas/spatial/trace_operator.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <cmath>     // std::sqrt
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

#include <tiny_obj_loader.h>

namespace atlas::geometry {

/* ====================================================================== */
/* TriangleMesh<T>                                                         */
/* ====================================================================== */

template <typename T>
TriangleMesh<T>::TriangleMesh(const HostBuffer<TriangleContainer4<T>>& triangles_) noexcept
    : triangles(triangles_) {
    // Construct from an existing triangle buffer (copy).
    //
    // Important:
    // - A BVH is used to accelerate ray tracing (TraceOperator path).
    // - We ensure the BVH object exists and immediately build it so that
    //   the mesh is ready for tracing as soon as construction finishes.
    ensure_bvh();
    build_bvh();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept
    : triangles(std::move(triangles_)) {
    // Construct from an existing triangle buffer (move).
    //
    // Rationale:
    // - Moving avoids copying potentially large mesh data.
    // - As above, build BVH eagerly for predictable runtime behavior.
    ensure_bvh();
    build_bvh();
}

template <typename T>
typename TriangleMesh<T>::Builder
TriangleMesh<T>::builder() noexcept {
    // Builder entry point.
    //
    // Usage:
    //   auto mesh = TriangleMesh<float>::builder()
    //                  .load_from_obj("model.obj")
    //                  .build();
    return Builder {};
}

template <typename T>
void
TriangleMesh<T>::set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_) {
    // Replace triangle storage and rebuild the BVH.
    //
    // We do the full rebuild because:
    // - Triangles are the BVH primitives; any change invalidates the tree.
    triangles = triangles_;
    query_cache_built = false;
    ensure_bvh();
    build_bvh();
}

/* ---------------------------------------------------------------------- */
/* BVH management                                                          */
/* ---------------------------------------------------------------------- */

template <typename T>
void
TriangleMesh<T>::ensure_bvh() noexcept {
    // Ensure the BVH pointer is initialized.
    //
    // Why separate from build_bvh()?
    // - Some code paths may want to guarantee the BVH object exists without
    //   necessarily building (e.g., after default construction).
    if (!_bvh) _bvh = atlas::make_host_shared<SAHBVH<T>>();
}

template <typename T>
void
TriangleMesh<T>::build_bvh() {
    // Build or rebuild the BVH over the current triangles.
    //
    // Behavior:
    // - If no BVH is present, do nothing (defensive).
    // - If triangles are empty, mark "not built" and return.
    // - Otherwise, call into SAHBVH<T>::build(...) and mark built.
    if (!_bvh) return;

    if (triangles.empty()) {
        bvh_built = false;
        return;
    }

    // SAH BVH build:
    // - Typically uses Surface Area Heuristic to create a good split tree.
    // - The build cost is higher than a simple median split, but tracing is
    //   usually faster afterwards.
    _bvh->build(triangles);

    bvh_built = true;
}

template <typename T>
void
TriangleMesh<T>::ensure_query_cache() const {
    if (query_cache_built) return;
    rebuild_query_cache();
}

template <typename T>
void
TriangleMesh<T>::rebuild_query_cache() const {
    const std::size_t count = triangles.size();
    _query_vertices.resize(count * 3);
    _query_indices.resize(count * 3);

    for (std::size_t t = 0; t < count; ++t) {
        const auto& tri         = triangles[t];
        const std::size_t base  = t * 3;

        _query_vertices[base + 0] = tri.a();
        _query_vertices[base + 1] = tri.b();
        _query_vertices[base + 2] = tri.c();

        _query_indices[base + 0] = static_cast<int>(base + 0);
        _query_indices[base + 1] = static_cast<int>(base + 1);
        _query_indices[base + 2] = static_cast<int>(base + 2);
    }

    query_cache_built = true;
}

/* ---------------------------------------------------------------------- */
/* Operators: Trace / Query                                                */
/* ---------------------------------------------------------------------- */

template <typename T>
TraceOperator<T>
TriangleMesh<T>::make_trace_operator() const {
    // Produce a type-erased TraceOperator<T> for ray intersection.
    //
    // If BVH is not available or not built, we return a default BVH operator.
    // That default should behave safely (typically "no hit") rather than crash.
    if (!_bvh || !bvh_built) {
        return TraceOperator<T>(atlas::spatial::BvhTraceOperator<T> {});
    }

    // Return BVH's own trace operator (fast path).
    return TraceOperator<T>(_bvh->make_trace_operator());
}

template <typename T>
QueryOperator<T>
TriangleMesh<T>::make_query_operator() const {
    // Build a QueryOperator view over a cached contiguous vertex/index representation.
    ensure_query_cache();

    TriangleMeshQueryOperator<T> op {};
    op.vertices       = _query_vertices.empty() ? nullptr : atlas::raw_pointer_cast(_query_vertices.data());
    op.indices        = _query_indices.empty() ? nullptr : atlas::raw_pointer_cast(_query_indices.data());
    op.triangle_count = static_cast<int>(triangles.size());
    return QueryOperator<T>(op);
}

/* ---------------------------------------------------------------------- */
/* OBJ loading                                                             */
/* ---------------------------------------------------------------------- */

template <typename T>
bool
TriangleMesh<T>::load_from_obj(const std::string& filename, const bool verbose) {
    // Load a triangle mesh from an OBJ file using tinyobjloader.
    //
    // Policy:
    // - We triangulate faces (config.triangulate = true).
    // - We ignore materials (mtl_search_path empty).
    // - We only use vertex positions for now.
    //
    // Output layout:
    // - Each triangle becomes one TriangleContainer4<T>:
    //     a(), b(), c() are vertices
    //     d() is used here to store the per-triangle normal
    //   (fits the "4 vectors" layout nicely and keeps per-tri data compact).
    triangles.clear();

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "";
    config.triangulate     = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config)) {
        // Parse failure:
        // - could be missing file, invalid OBJ, etc.
        return false;
    }

    const tinyobj::attrib_t attrib             = reader.GetAttrib();
    const std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

    // We need at least vertex positions.
    if (attrib.vertices.empty()) return false;

    const auto& v     = attrib.vertices; // flat float array: [x0,y0,z0, x1,y1,z1, ...]
    const auto vcount = v.size();        // number of floats (not vertices)

    for (const auto& shape : shapes) {
        const auto& idx = shape.mesh.indices;

        // Since triangulate=true, indices are already triangles in groups of 3,
        // but we still iterate in steps of 3 defensively.
        for (std::size_t f = 0; f + 2 < idx.size(); f += 3) {
            const int vi0 = idx[f + 0].vertex_index;
            const int vi1 = idx[f + 1].vertex_index;
            const int vi2 = idx[f + 2].vertex_index;

            // Negative vertex index indicates "missing" (per tinyobj rules).
            if (vi0 < 0 || vi1 < 0 || vi2 < 0) continue;

            // Convert vertex indices to offsets into the flat float array.
            const std::size_t o0 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi0);
            const std::size_t o1 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi1);
            const std::size_t o2 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi2);

            // Bounds check (o + 2 must be valid index into v).
            if (o0 + 2 >= vcount || o1 + 2 >= vcount || o2 + 2 >= vcount) continue;

            // Read positions and cast to T.
            const atlas::math::Vector<T, 3> a(
                static_cast<T>(v[o0 + 0]),
                static_cast<T>(v[o0 + 1]),
                static_cast<T>(v[o0 + 2]));
            const atlas::math::Vector<T, 3> b(
                static_cast<T>(v[o1 + 0]),
                static_cast<T>(v[o1 + 1]),
                static_cast<T>(v[o1 + 2]));
            const atlas::math::Vector<T, 3> c(
                static_cast<T>(v[o2 + 0]),
                static_cast<T>(v[o2 + 1]),
                static_cast<T>(v[o2 + 2]));

            // Pack into the container.
            TriangleContainer4<T> tc;
            tc.a() = a;
            tc.b() = b;
            tc.c() = c;

            // Compute per-triangle geometric normal.
            // - Normal is cross(b-a, c-a)
            // - We normalize it so it can be used directly for signed distance sign.
            atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
            const T n2                  = n.length_squared();

            if (n2 > T(0)) {
                n *= (T(1) / static_cast<T>(std::sqrt(n2)));
            } else {
                // Degenerate triangle: fall back to a default axis.
                n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
            }

    // Store normal in the 4th slot.
    tc.d() = n;

    triangles.push_back(tc);
        }
    }

    // If we loaded nothing, treat as failure.
    if (triangles.empty()) return false;

    query_cache_built = false;

    // Build acceleration structure for tracing.
    ensure_bvh();
    build_bvh();

    // (Optional) verbose is currently unused. If you want it:
    // - print triangle count, BVH stats, etc.
    (void)verbose;
    return true;
}

/* ---------------------------------------------------------------------- */
/* Closest-point / normal / SDF (host path)                                */
/* ---------------------------------------------------------------------- */

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
#if !defined(__CUDA_ARCH__)
    if (triangles.empty()) return p;

    // Host path delegates to TriangleMeshQueryOperator for consistent mesh semantics.
    return make_query_operator().closest_point(p);
#else
    // Find the closest point on the mesh to p by scanning all triangles.
    //
    // Complexity:
    // - O(Ntri) per query (no BVH used here).
    // - This is fine for small meshes / debugging, but for large meshes
    //   consider a BVH-accelerated closest-point query.
    if (triangles.empty()) return p;

    T best_d2                         = std::numeric_limits<T>::infinity();
    atlas::math::Vector<T, 3> best_cp = p;

    for (const auto& tc : triangles) {
        // Wire up a TriangleQueryOperator to the triangle's stored vertices.
        // Note: tc.d() is used as the normal pointer.
        TriangleQueryOperator<T> tri {};
        tri.a = &tc.a();
        tri.b = &tc.b();
        tri.c = &tc.c();
        tri.n = &tc.d();

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const atlas::math::Vector<T, 3> d  = cp - p;
        const T d2                         = d.length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
            best_cp = cp;
        }
    }

    return best_cp;
#endif
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
#if !defined(__CUDA_ARCH__)
    if (triangles.empty()) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    // Host path delegates to TriangleMeshQueryOperator for consistent mesh semantics.
    return make_query_operator().closest_normal(p);
#else
    // Return the normal of the triangle (or feature) that is closest to p.
    //
    // Strategy:
    // - Find the triangle with minimal squared distance from p to its closest point.
    // - Return that triangle's closest_normal().
    if (triangles.empty()) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    T best_d2 = std::numeric_limits<T>::infinity();
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(1));

    for (const auto& tc : triangles) {
        TriangleQueryOperator<T> tri {};
        tri.a = &tc.a();
        tri.b = &tc.b();
        tri.c = &tc.c();
        tri.n = &tc.d();

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const atlas::math::Vector<T, 3> d  = cp - p;
        const T d2                         = d.length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
            best_n  = tri.closest_normal(p);
        }
    }

    return best_n;
#endif
}

template <typename T>
T
TriangleMesh<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
#if !defined(__CUDA_ARCH__)
    if (triangles.empty()) return std::numeric_limits<T>::infinity();

    // Host path uses the mesh query operator, including winding-based sign.
    return make_query_operator().signed_distance(p);
#else
    // Signed distance to the mesh (triangle soup) using nearest triangle.
    //
    // Steps:
    //  1) Find nearest triangle by closest-point distance.
    //  2) Let dist = |p - cp|.
    //  3) Determine sign using dot(n, p - cp):
    //        >= 0 => outside (positive)
    //         < 0 => inside (negative) relative to the oriented triangle normal
    //
    // Caveat:
    // - For an open mesh, "inside/outside" is not globally meaningful.
    // - This provides a *local* signed distance relative to the nearest surface.
    if (triangles.empty()) return std::numeric_limits<T>::infinity();

    T best_d2                         = std::numeric_limits<T>::infinity();
    atlas::math::Vector<T, 3> best_cp = p;
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(1));

    for (const auto& tc : triangles) {
        TriangleQueryOperator<T> tri {};
        tri.a = &tc.a();
        tri.b = &tc.b();
        tri.c = &tc.c();
        tri.n = &tc.d();

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const atlas::math::Vector<T, 3> d  = p - cp; // vector from surface to p
        const T d2                         = d.length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
            best_cp = cp;
            best_n  = tri.closest_normal(p);
        }
    }

    const atlas::math::Vector<T, 3> v = p - best_cp;
    const T dist                      = static_cast<T>(std::sqrt(best_d2));

    // Sign convention:
    // - If p is on the same side as the normal, return positive.
    // - Otherwise negative.
    const T s = (best_n.dot(v) >= T(0)) ? T(1) : T(-1);

    return s * dist;
#endif
}

template <typename T>
bool
TriangleMesh<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    if (triangles.empty()) return false;
    // Use the mesh query operator so inside classification matches winding-based semantics.
    return make_query_operator().is_inside(p, tolerance);
}

template <typename T>
bool
TriangleMesh<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    if (triangles.empty()) return false;
    // Surface classification follows the mesh query-operator distance convention.
    return make_query_operator().is_on_surface(p, tolerance);
}

/* ---------------------------------------------------------------------- */
/* Centroid / Bounds / Validity                                            */
/* ---------------------------------------------------------------------- */

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::centroid() const noexcept {
#if !defined(__CUDA_ARCH__)
    if (triangles.empty()) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // Reuse the query-operator centroid to keep mesh query behavior centralized.
    return make_query_operator().centroid();
#else
    // Compute an average of triangle centroids (uniform per-triangle weighting).
    //
    // Note:
    // - This is NOT an area-weighted centroid.
    // - For a better geometric centroid, weight each triangle by its area.
    if (triangles.empty()) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    atlas::math::Vector<T, 3> acc(T(0), T(0), T(0));
    const T inv = T(1) / static_cast<T>(triangles.size());

    for (const auto& tc : triangles) {
        const atlas::math::Vector<T, 3> c = (tc.a() + tc.b() + tc.c()) * (T(1) / T(3));
        acc += c;
    }

    return acc * inv;
#endif
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMesh<T>::bound() const noexcept {
#if !defined(__CUDA_ARCH__)
    if (triangles.empty()) return atlas::spatial::AxisAlignedBoundingBox<T>();

    // Reuse the query-operator bound to keep mesh query behavior centralized.
    return make_query_operator().bound();
#else
    // Compute the axis-aligned bounding box over all vertices.
    //
    // Implementation:
    // - Initialize with the first vertex, then component-wise min/max accumulate.
    if (triangles.empty()) return atlas::spatial::AxisAlignedBoundingBox<T>();

    atlas::math::Vector<T, 3> lo = triangles[0].a();
    atlas::math::Vector<T, 3> hi = triangles[0].a();

    for (const auto& tc : triangles) {
        lo = atlas::math::cmin(lo, tc.a());
        lo = atlas::math::cmin(lo, tc.b());
        lo = atlas::math::cmin(lo, tc.c());

        hi = atlas::math::cmax(hi, tc.a());
        hi = atlas::math::cmax(hi, tc.b());
        hi = atlas::math::cmax(hi, tc.c());
    }

    return atlas::spatial::AxisAlignedBoundingBox<T>(lo, hi);
#endif
}

template <typename T>
bool
TriangleMesh<T>::is_valid() const noexcept {
    // Validate that the mesh contains only non-degenerate triangles.
    //
    // Check:
    // - triangles must be non-empty
    // - each triangle must have non-zero area (cross product magnitude > 0)
    if (triangles.empty()) return false;

    for (const auto& tc : triangles) {
        const atlas::math::Vector<T, 3> a = tc.a();
        const atlas::math::Vector<T, 3> b = tc.b();
        const atlas::math::Vector<T, 3> c = tc.c();

        const atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        if (!(n.length_squared() > T(0))) return false;
    }

    return true;
}

template <typename T>
GeometryType
TriangleMesh<T>::type() const noexcept {
    // Return the geometry type tag for this class.
    return GeometryType::TriangleMesh;
}

/* ====================================================================== */
/* TriangleMesh<T>::Builder                                                */
/* ====================================================================== */

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(const HostBuffer<TriangleContainer4<T>>& ts) {
    // Copy triangles into builder staging storage.
    _triangles = ts;
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(HostBuffer<TriangleContainer4<T>>&& ts) {
    // Move triangles into builder staging storage.
    _triangles = std::move(ts);
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::load_from_obj(const std::string& filename, const bool verbose) {
    // Convenience:
    // - Use TriangleMesh<T>::load_from_obj(...) to parse and triangulate
    // - Then steal the resulting triangle buffer into the builder.
    TriangleMesh<T> tmp {};
    const bool ok = tmp.load_from_obj(filename, verbose);

    if (!ok) {
        throw std::runtime_error("TriangleMesh::Builder: failed to load OBJ.");
    }

    _triangles = std::move(tmp.triangles);
    return *this;
}

template <typename T>
void
TriangleMesh<T>::Builder::validate() const {
    // Validate builder state before building a TriangleMesh<T>.
    // Policy:
    // - We require at least one triangle (caller must provide geometry).
    // - We do not validate individual triangle geometry here (e.g., zero-area).
    if (_triangles.empty()) {
        atlas::logger::error()
            << "TriangleMesh::Builder validation failed: no triangles provided.";
        throw std::runtime_error("TriangleMesh::Builder validation failed: no triangles provided.");
    }
}

template <typename T>
TriangleMesh<T>
TriangleMesh<T>::Builder::build() const {
    // Build a TriangleMesh<T> from the staged triangles.
    //
    // Policy:
    // - Reject empty meshes (caller must supply geometry).
    validate();

    TriangleMesh<T> m {};

    // Copy triangles into the mesh instance.
    // If you want to avoid a copy, add a rvalue-qualified build() overload
    // that moves _triangles.
    m.triangles = _triangles;

    // Build acceleration structure for tracing.
    m.ensure_bvh();
    m.build_bvh();

    return m;
}

template <typename T>
atlas::host_shared_ptr<TriangleMesh<T>>
TriangleMesh<T>::Builder::make_host_shared() const {
    // Allocate a shared-owned mesh on the host, moving the built value object.
    auto m = build();
    return atlas::make_host_shared<TriangleMesh<T>>(std::move(m));
}

} // namespace atlas::geometry
