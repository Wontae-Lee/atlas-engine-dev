#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tiny_obj_loader.h>
#include <utility>

namespace atlas::geometry {

template <typename T>
TriangleMesh<T>::TriangleMesh(const HostBuffer<TriangleContainer4<T>>& triangles_) noexcept
    : triangles(triangles_) {
    // Allocate the BVH container before building acceleration data.
    ensure_bvh();

    // Build the BVH over the provided triangle buffer.
    build_bvh();

    // Build the flattened vertex/index cache used by geometry queries.
    ensure_query_cache();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept
    : triangles(std::move(triangles_)) {
    // Allocate the BVH container before building acceleration data.
    ensure_bvh();

    // Build the BVH over the moved triangle buffer.
    build_bvh();

    // Build the flattened vertex/index cache used by geometry queries.
    ensure_query_cache();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(const TriangleMesh& other)
    : triangles(other.triangles)
    , _bvh(other._bvh)
    , _query_vertices(other._query_vertices)
    , _query_indices(other._query_indices)
    , bvh_built(other.bvh_built)
    , query_cache_built(other.query_cache_built) {
    // Rebind the operator pointers to this copied mesh's cache and BVH state.
    update_operator();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(TriangleMesh&& other) noexcept
    : triangles(std::move(other.triangles))
    , _bvh(std::move(other._bvh))
    , _query_vertices(std::move(other._query_vertices))
    , _query_indices(std::move(other._query_indices))
    , bvh_built(other.bvh_built)
    , query_cache_built(other.query_cache_built) {
    // Rebind this mesh after taking ownership of the moved resources.
    update_operator();

    // Mark the moved-from mesh as no longer owning valid acceleration/query data.
    other.bvh_built         = false;
    other.query_cache_built = false;

    // Keep the moved-from operator in a safe, empty state.
    other.update_operator();
}

template <typename T>
TriangleMesh<T>&
TriangleMesh<T>::operator=(const TriangleMesh& other) {
    // Avoid unnecessary copying and rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    triangles         = other.triangles;
    _bvh              = other._bvh;
    _query_vertices   = other._query_vertices;
    _query_indices    = other._query_indices;
    bvh_built         = other.bvh_built;
    query_cache_built = other.query_cache_built;

    // Rebind the operator pointers to this assigned mesh's local storage.
    update_operator();

    return *this;
}

template <typename T>
TriangleMesh<T>&
TriangleMesh<T>::operator=(TriangleMesh&& other) noexcept {
    // Avoid self move-assignment.
    if (this == &other) {
        return *this;
    }

    triangles         = std::move(other.triangles);
    _bvh              = std::move(other._bvh);
    _query_vertices   = std::move(other._query_vertices);
    _query_indices    = std::move(other._query_indices);
    bvh_built         = other.bvh_built;
    query_cache_built = other.query_cache_built;

    // Rebind this mesh after taking ownership of moved resources.
    update_operator();

    // Mark the moved-from mesh as empty and rebuild its operator state.
    other.bvh_built         = false;
    other.query_cache_built = false;
    other.update_operator();

    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder
TriangleMesh<T>::builder() noexcept {
    // Return a fresh builder for fluent triangle-mesh construction.
    return Builder {};
}

template <typename T>
void
TriangleMesh<T>::set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_) {
    // Replace the mesh triangle storage.
    triangles = triangles_;

    // Mark the query cache dirty because the triangle data changed.
    query_cache_built = false;

    // Rebuild acceleration and query data from the new triangles.
    ensure_bvh();
    build_bvh();
    ensure_query_cache();
}

template <typename T>
void
TriangleMesh<T>::ensure_bvh() noexcept {
    // Lazily allocate the BVH object when it does not exist yet.
    if (!_bvh) {
        _bvh = atlas::make_host_shared<SAHBVH<T>>();
    }
}

template <typename T>
void
TriangleMesh<T>::build_bvh() {
    // Without a BVH object, no acceleration structure can be built.
    if (!_bvh) {
        return;
    }

    // Empty meshes have no valid BVH.
    if (triangles.empty()) {
        bvh_built = false;
        return;
    }

    // Build the surface-area-heuristic BVH over the triangle container buffer.
    _bvh->build(triangles);
    bvh_built = true;
}

template <typename T>
void
TriangleMesh<T>::ensure_query_cache() const {
    // Avoid rebuilding when the flattened query data is already valid.
    if (query_cache_built) {
        return;
    }

    // Rebuild flattened vertices and indices for operator-side queries.
    rebuild_query_cache();
}

template <typename T>
void
TriangleMesh<T>::rebuild_query_cache() const {
    const std::size_t count = triangles.size();

    // Store three explicit vertices and three indices per triangle.
    _query_vertices.resize(count * 3);
    _query_indices.resize(count * 3);

    for (std::size_t t = 0; t < count; ++t) {
        const auto& tri        = triangles[t];
        const std::size_t base = t * 3;

        // Flatten triangle vertices into a contiguous query buffer.
        _query_vertices[base + 0] = tri.a();
        _query_vertices[base + 1] = tri.b();
        _query_vertices[base + 2] = tri.c();

        // Each triangle uses its own three sequential flattened vertices.
        _query_indices[base + 0] = static_cast<int>(base + 0);
        _query_indices[base + 1] = static_cast<int>(base + 1);
        _query_indices[base + 2] = static_cast<int>(base + 2);
    }

    query_cache_built = true;

    // Refresh operator pointers after the cache buffers have been resized.
    update_operator();
}

template <typename T>
void
TriangleMesh<T>::update_operator() const {
    // Bind the operator to flattened vertex and index buffers when available.
    _operator.vertices       = _query_vertices.empty() ? nullptr : atlas::raw_pointer_cast(_query_vertices.data());
    _operator.indices        = _query_indices.empty() ? nullptr : atlas::raw_pointer_cast(_query_indices.data());
    _operator.triangle_count = static_cast<int>(triangles.size());

    if (_bvh && bvh_built) {
        // Copy BVH operator pointers so trace queries can traverse the acceleration structure.
        const auto bvh_op     = _bvh->make_geometry_operator();
        _operator.bvh_nodes   = bvh_op.bvh_nodes;
        _operator.bvh_indices = bvh_op.bvh_indices;
        _operator.bvh_tris    = bvh_op.bvh_tris;
        _operator.bvh_root    = bvh_op.bvh_root;
    } else {
        // Clear BVH pointers when no valid acceleration structure exists.
        _operator.bvh_nodes   = nullptr;
        _operator.bvh_indices = nullptr;
        _operator.bvh_tris    = nullptr;
        _operator.bvh_root    = -1;
    }
}

template <typename T>
GeometryOperator<T>
TriangleMesh<T>::make_geometry_operator() const {
    // Ensure flattened query buffers are ready before exposing the operator.
    ensure_query_cache();

    // Return a type-erased geometry operator backed by this mesh operator.
    return GeometryOperator<T>(_operator);
}

template <typename T>
bool
TriangleMesh<T>::load_from_obj(const std::string& filename, const bool verbose) {
    // Clear current mesh data before loading new OBJ geometry.
    triangles.clear();
    query_cache_built = false;

    tinyobj::ObjReaderConfig config;

    // Disable material search path and force triangulation for mesh consistency.
    config.mtl_search_path = "";
    config.triangulate     = true;

    tinyobj::ObjReader reader;

    // Parse the OBJ file and reject on loader failure.
    if (!reader.ParseFromFile(filename, config)) {
        return false;
    }

    const tinyobj::attrib_t attrib             = reader.GetAttrib();
    const std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

    // A mesh without vertex positions cannot produce triangles.
    if (attrib.vertices.empty()) {
        return false;
    }

    const auto& v     = attrib.vertices;
    const auto vcount = v.size();

    for (const auto& shape : shapes) {
        const auto& idx = shape.mesh.indices;

        // Iterate over triangulated faces in groups of three indices.
        for (std::size_t f = 0; f + 2 < idx.size(); f += 3) {
            const int vi0 = idx[f + 0].vertex_index;
            const int vi1 = idx[f + 1].vertex_index;
            const int vi2 = idx[f + 2].vertex_index;

            // Skip faces with missing vertex indices.
            if (vi0 < 0 || vi1 < 0 || vi2 < 0) {
                continue;
            }

            // Convert vertex indices into offsets within the flat xyz vertex array.
            const std::size_t o0 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi0);
            const std::size_t o1 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi1);
            const std::size_t o2 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi2);

            // Skip faces whose indices would read outside the vertex array.
            if (o0 + 2 >= vcount || o1 + 2 >= vcount || o2 + 2 >= vcount) {
                continue;
            }

            // Convert OBJ vertex positions into Atlas vector coordinates.
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

            TriangleContainer4<T> tc;

            // Store triangle vertices in the container.
            tc.a() = a;
            tc.b() = b;
            tc.c() = c;

            // Compute and store a normalized triangle normal in the fourth slot.
            atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
            const T n2                  = n.length_squared();

            if (n2 > T(0)) {
                n *= (T(1) / static_cast<T>(std::sqrt(n2)));
            } else {
                // Degenerate faces receive a deterministic fallback normal.
                n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
            }

            tc.d() = n;

            // Append the loaded triangle to the mesh.
            triangles.push_back(tc);
        }
    }

    // Reject files that did not produce any valid triangle.
    if (triangles.empty()) {
        return false;
    }

    // Rebuild acceleration and query data after loading.
    query_cache_built = false;
    ensure_bvh();
    build_bvh();
    ensure_query_cache();

    // Parameter is currently reserved for optional diagnostic output.
    (void)verbose;

    return true;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-point queries to the bound mesh operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-normal queries to the bound mesh operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
TriangleMesh<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate signed-distance queries to the bound mesh operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
TriangleMesh<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate containment checks to the bound mesh operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
TriangleMesh<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-membership checks to the bound mesh operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::centroid() const noexcept {
    // Delegate centroid computation to the bound mesh operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMesh<T>::bound() const noexcept {
    // Delegate bounding-box construction to the bound mesh operator.
    return _operator.bound();
}

template <typename T>
bool
TriangleMesh<T>::is_valid() const noexcept {
    // Delegate validity checks to the bound mesh operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
TriangleMesh<T>::type() const noexcept {
    // Identify this geometry as a triangle mesh.
    return GeometryType::TriangleMesh;
}

template <typename T>
TriangleMesh<T>
TriangleMesh<T>::Builder::build() const {
    // Validate builder state before constructing the final mesh.
    validate();

    TriangleMesh<T> m {};
    m.triangles = _triangles;

    // Build the required acceleration and query data after assigning triangles.
    m.ensure_bvh();
    m.build_bvh();
    m.ensure_query_cache();

    return m;
}

template <typename T>
atlas::host_shared_ptr<TriangleMesh<T>>
TriangleMesh<T>::Builder::make_host_shared() const {
    // Build a validated mesh and store it in host-managed shared ownership.
    auto m = build();
    return atlas::make_host_shared<TriangleMesh<T>>(std::move(m));
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(const HostBuffer<TriangleContainer4<T>>& ts) {
    // Copy triangle data into the builder.
    _triangles = ts;
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(HostBuffer<TriangleContainer4<T>>&& ts) {
    // Move triangle data into the builder.
    _triangles = std::move(ts);
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::load_from_obj(const std::string& filename, const bool verbose) {
    TriangleMesh<T> tmp {};

    // Reuse TriangleMesh loading logic to populate temporary mesh triangles.
    const bool ok = tmp.load_from_obj(filename, verbose);

    if (!ok) {
        throw std::runtime_error("TriangleMesh::Builder: failed to load OBJ.");
    }

    // Move the loaded triangles into the builder state.
    _triangles = std::move(tmp.triangles);

    return *this;
}

template <typename T>
void
TriangleMesh<T>::Builder::validate() const {
    // A mesh must contain at least one triangle.
    if (_triangles.empty()) {
        throw std::runtime_error("TriangleMesh::Builder validation failed: no triangles provided.");
    }
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::solid_angle(const atlas::math::Vector<T, 3>& p,
                                             const atlas::math::Vector<T, 3>& a,
                                             const atlas::math::Vector<T, 3>& b,
                                             const atlas::math::Vector<T, 3>& c) const noexcept {
    // Express all triangle vertices relative to the query point.
    const atlas::math::Vector<T, 3> va = a - p;
    const atlas::math::Vector<T, 3> vb = b - p;
    const atlas::math::Vector<T, 3> vc = c - p;

    const T la = va.length();
    const T lb = vb.length();
    const T lc = vc.length();

    // Avoid singular solid-angle evaluation when the point coincides with a vertex.
    if (la <= std::numeric_limits<T>::epsilon()
        || lb <= std::numeric_limits<T>::epsilon()
        || lc <= std::numeric_limits<T>::epsilon()) {
        return T(0);
    }

    // Oriented tetrahedral volume term for the solid-angle numerator.
    const T numerator = va.dot(atlas::math::cross(vb, vc));

    // Robust denominator for the triangle solid angle formula.
    const T denominator = la * lb * lc
        + va.dot(vb) * lc
        + vb.dot(vc) * la
        + vc.dot(va) * lb;

    return T(2) * std::atan2(numerator, denominator);
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::winding_number(const atlas::math::Vector<T, 3>& p) const noexcept {
    T solid_angle_sum = T(0);

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve the three flattened vertex indices for the current triangle.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Accumulate the oriented solid angle contribution.
        solid_angle_sum += solid_angle(p, a, b, c);
    }

    const T four_pi = T(4) * std::acos(T(-1));

    // Normalize total solid angle to get the winding number.
    return solid_angle_sum / four_pi;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Without valid mesh data, there is no meaningful projection target.
    if (!is_valid()) {
        return p;
    }

    T best_d2                         = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_cp = p;

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve the current triangle vertices from the flattened buffers.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Compute a per-triangle normal for the temporary triangle operator.
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();

        if (n2 > T(0)) {
            n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        } else {
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
        }

        // Bind the temporary triangle operator to the current triangle.
        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        // Keep the closest projection found across all triangles.
        if (d2 < best_d2) {
            best_d2 = d2;
            best_cp = cp;
        }
    }

    return best_cp;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid mesh data falls back to a deterministic normal.
    if (!is_valid()) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(1));
    }

    T best_d2 = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(1));

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve the current triangle vertices from the flattened buffers.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Compute a normalized per-triangle normal.
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();

        if (n2 > T(0)) {
            n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        } else {
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
        }

        // Bind the temporary triangle operator to the current triangle.
        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        // Use the normal of the triangle that owns the closest point.
        if (d2 < best_d2) {
            best_d2 = d2;
            best_n  = tri.closest_normal(p);
        }
    }

    return best_n;
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid mesh data is treated as infinitely far away.
    if (!is_valid()) {
        return std::numeric_limits<T>::infinity();
    }

    T best_d2 = std::numeric_limits<T>::max();

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve the current triangle vertices from the flattened buffers.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Compute a normalized per-triangle normal for closest-point evaluation.
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();

        if (n2 > T(0)) {
            n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        } else {
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
        }

        // Bind the temporary triangle operator to the current triangle.
        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        // Track the shortest squared distance to the mesh surface.
        if (d2 < best_d2) {
            best_d2 = d2;
        }
    }

    const T dist = static_cast<T>(std::sqrt(best_d2));

    // Points numerically on the mesh surface have zero signed distance.
    if (dist <= std::numeric_limits<T>::epsilon()) {
        return T(0);
    }

    // Winding number determines whether the point is inside a closed mesh.
    const T winding = winding_number(p);

    return (std::abs(winding) > T(0.5)) ? -dist : dist;
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid mesh data cannot contain any point.
    if (!is_valid()) {
        return false;
    }

    T best_d2 = std::numeric_limits<T>::max();

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve the current triangle vertices from the flattened buffers.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Compute a normalized per-triangle normal for closest-point evaluation.
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();

        if (n2 > T(0)) {
            n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        } else {
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
        }

        // Bind the temporary triangle operator to the current triangle.
        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        // Track the shortest squared distance to the mesh surface.
        if (d2 < best_d2) {
            best_d2 = d2;
        }
    }

    const T dist      = static_cast<T>(std::sqrt(best_d2));
    const T winding   = winding_number(p);
    const bool inside = std::abs(winding) > T(0.5);

    if (inside) {
        // Non-negative tolerance accepts all points classified inside.
        if (tolerance >= T(0)) {
            return true;
        }

        // Negative tolerance shrinks the accepted interior away from the surface.
        return dist >= -tolerance;
    }

    // Outside points cannot be accepted with negative tolerance.
    if (tolerance < T(0)) {
        return false;
    }

    // Positive tolerance accepts outside points close to the mesh surface.
    return dist <= tolerance;
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Surface membership is measured by the absolute signed distance.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::centroid() const noexcept {
    // Invalid mesh data falls back to the origin as a neutral centroid.
    if (!is_valid()) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    atlas::math::Vector<T, 3> sum(T(0), T(0), T(0));

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve the current triangle vertices from the flattened buffers.
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Accumulate the centroid of each triangle.
        sum += (a + b + c) * (T(1) / T(3));
    }

    // Return the average of triangle centroids.
    return sum * (T(1) / static_cast<T>(triangle_count));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMeshGeometryOperator<T>::bound() const noexcept {
    // Invalid mesh data returns an empty/default bounding box.
    if (!is_valid()) {
        return atlas::spatial::AxisAlignedBoundingBox<T>();
    }

    // Initialize min and max corners from the first referenced vertex.
    const int i0                 = indices[0];
    atlas::math::Vector<T, 3> mn = vertices[i0];
    atlas::math::Vector<T, 3> mx = vertices[i0];

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve all vertices of the current triangle.
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

        // Expand the AABB by each triangle vertex.
        mn = atlas::math::cmin(mn, vertices[j0]);
        mn = atlas::math::cmin(mn, vertices[j1]);
        mn = atlas::math::cmin(mn, vertices[j2]);

        mx = atlas::math::cmax(mx, vertices[j0]);
        mx = atlas::math::cmax(mx, vertices[j1]);
        mx = atlas::math::cmax(mx, vertices[j2]);
    }

    return atlas::spatial::AxisAlignedBoundingBox<T>(mn, mx);
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_valid() const noexcept {
    // Vertex and index buffers must be bound.
    if (!vertices || !indices) {
        return false;
    }

    // A mesh must contain at least one triangle.
    if (triangle_count <= 0) {
        return false;
    }

    for (int t = 0; t < triangle_count; ++t) {
        // Resolve triangle vertex indices.
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

        // Reject degenerate triangles with near-zero area.
        const auto ab = vertices[j1] - vertices[j0];
        const auto ac = vertices[j2] - vertices[j0];
        const auto n  = atlas::math::cross(ab, ac);

        if (!(n.length_squared() > static_cast<T>(atlas::eps))) {
            return false;
        }
    }

    return true;
}

template <typename T>
HitSurface<T>
TriangleMeshGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    HitSurface<T> out {};

    // BVH traversal requires valid BVH buffers and a valid root index.
    if (!bvh_nodes || !bvh_indices || !bvh_tris || bvh_root < 0) {
        return out;
    }

    // Track the nearest triangle hit found during BVH traversal.
    T best_t = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_p {};
    atlas::math::Vector<T, 3> best_n {};
    bool found = false;

    // Use a fixed-size traversal stack for iterative BVH traversal.
    int stack[64];
    int sp      = 0;
    stack[sp++] = bvh_root;

    while (sp) {
        // Pop the next BVH node from the traversal stack.
        const int ni                         = stack[--sp];
        const atlas::spatial::BVHNode<T>& nd = bvh_nodes[ni];

        // Reject nodes whose bounding boxes are missed or farther than the best hit.
        const auto hit = nd.bounds.trace(r);
        if (!hit.is_intersecting || hit.enter > best_t) {
            continue;
        }

        if (nd.is_leaf) {
            TriangleGeometryOperator<T> tri_op {};

            ATLAS_UNROLL
            for (int k = 0; k < nd.count; ++k) {
                // Resolve the primitive stored in this BVH leaf.
                const int pid                    = bvh_indices[nd.start + k];
                const TriangleContainer4<T>& tri = bvh_tris[pid];

                // Bind a temporary triangle operator to the BVH triangle data.
                tri_op.a = &tri.a();
                tri_op.b = &tri.b();
                tri_op.c = &tri.c();
                tri_op.n = &tri.d();

                const HitSurface<T> h = tri_op.trace(r);

                // Keep the closest valid triangle hit.
                if (h.is_intersecting && h.distance < best_t) {
                    best_t = h.distance;
                    best_p = h.point;
                    best_n = h.normal;
                    found  = true;
                }
            }
        } else {
            // Push child nodes onto the stack, clamping to the fixed stack capacity.
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
        // Populate the hit result with the nearest triangle intersection.
        out.is_intersecting = true;
        out.distance        = best_t;
        out.point           = best_p;
        out.normal          = best_n;
    }

    return out;
}

template <typename T>
HitSurface<T>
TriangleMeshGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

} // namespace atlas::geometry