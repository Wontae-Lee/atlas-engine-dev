#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <tiny_obj_loader.h>

namespace atlas::geometry {

template <typename T>
TriangleMesh<T>::TriangleMesh(const HostBuffer<TriangleContainer4<T>>& triangles_) noexcept
    : triangles(triangles_) {
    // Copy the input triangle container data into mesh-owned storage.
    //
    // After the geometric data is available, immediately prepare all
    // acceleration and query-side caches so the mesh is ready for spatial queries.
    ensure_bvh();
    build_bvh();
    ensure_query_cache();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept
    : triangles(std::move(triangles_)) {
    // Move the input triangle container data into mesh-owned storage to avoid
    // an extra copy when the caller provides rvalue triangle data.
    //
    // As in the copy-based constructor, immediately build all dependent caches.
    ensure_bvh();
    build_bvh();
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
    // Copy all geometric storage, optional BVH handle, and flattened query caches
    // from the source mesh.
    //
    // The operator stores raw pointers into mesh-owned cache arrays, so it must be
    // refreshed after copying.
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
    // Move all heavy storage from the source mesh into this mesh.
    //
    // This includes:
    // - triangle storage,
    // - BVH handle,
    // - flattened vertex/index query caches,
    // - cache-validity flags.
    update_operator();

    // The moved-from object should no longer claim that its acceleration or
    // query caches are valid.
    other.bvh_built         = false;
    other.query_cache_built = false;

    // Refresh the moved-from object's operator so its raw pointers reflect
    // its post-move state.
    other.update_operator();
}

template <typename T>
TriangleMesh<T>&
TriangleMesh<T>::operator=(const TriangleMesh& other) {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy all persistent mesh state from the source.
    triangles         = other.triangles;
    _bvh              = other._bvh;
    _query_vertices   = other._query_vertices;
    _query_indices    = other._query_indices;
    bvh_built         = other.bvh_built;
    query_cache_built = other.query_cache_built;

    // Rebind operator raw pointers to this mesh's storage.
    update_operator();
    return *this;
}

template <typename T>
TriangleMesh<T>&
TriangleMesh<T>::operator=(TriangleMesh&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move all mesh-owned state from the source.
    triangles         = std::move(other.triangles);
    _bvh              = std::move(other._bvh);
    _query_vertices   = std::move(other._query_vertices);
    _query_indices    = std::move(other._query_indices);
    bvh_built         = other.bvh_built;
    query_cache_built = other.query_cache_built;

    // Refresh this mesh's operator after ownership transfer.
    update_operator();

    // Invalidate the moved-from object's cache-status flags.
    other.bvh_built         = false;
    other.query_cache_built = false;

    // Refresh the moved-from object's operator as well.
    other.update_operator();
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder
TriangleMesh<T>::builder() noexcept {
    // Return a fresh builder object for staged triangle-mesh construction.
    return Builder {};
}

template <typename T>
void
TriangleMesh<T>::set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_) {
    // Replace the current triangle storage with a new host-side triangle set.
    triangles = triangles_;

    // Any cached flattened query arrays are now stale because the underlying
    // triangle topology/geometry changed.
    query_cache_built = false;

    // Ensure the BVH object exists, rebuild the acceleration structure from
    // the new triangles, then rebuild the query cache if needed.
    ensure_bvh();
    build_bvh();
    ensure_query_cache();
}

template <typename T>
void
TriangleMesh<T>::ensure_bvh() noexcept {
    // Lazily allocate the BVH object only when it is actually needed.
    //
    // This keeps default / empty mesh construction lightweight until
    // acceleration data becomes necessary.
    if (!_bvh) _bvh = atlas::make_host_shared<SAHBVH<T>>();
}

template <typename T>
void
TriangleMesh<T>::build_bvh() {
    // If there is no BVH object, nothing can be built.
    if (!_bvh) return;

    if (triangles.empty()) {
        // An empty mesh has no valid acceleration structure.
        bvh_built = false;
        return;
    }

    // Build the SAH-based BVH from the current triangle storage.
    _bvh->build(triangles);

    // Mark the acceleration structure as valid.
    bvh_built = true;
}

template <typename T>
void
TriangleMesh<T>::ensure_query_cache() const {
    // Rebuild the flattened query cache only when it is not already valid.
    if (query_cache_built) return;
    rebuild_query_cache();
}

template <typename T>
void
TriangleMesh<T>::rebuild_query_cache() const {
    // Flatten the mesh triangle storage into:
    // - a contiguous vertex array with 3 vertices per triangle
    // - a matching contiguous index array with 3 indices per triangle
    //
    // This cache is designed for query operators that expect a compact
    // triangle soup representation.
    const std::size_t count = triangles.size();
    _query_vertices.resize(count * 3);
    _query_indices.resize(count * 3);

    for (std::size_t t = 0; t < count; ++t) {
        const auto& tri        = triangles[t];
        const std::size_t base = t * 3;

        // Store the triangle vertices in contiguous order.
        _query_vertices[base + 0] = tri.a();
        _query_vertices[base + 1] = tri.b();
        _query_vertices[base + 2] = tri.c();

        // Build a direct 0..N-1 index mapping into the flattened vertex array.
        _query_indices[base + 0] = static_cast<int>(base + 0);
        _query_indices[base + 1] = static_cast<int>(base + 1);
        _query_indices[base + 2] = static_cast<int>(base + 2);
    }

    // Mark the flattened query cache as valid and refresh operator pointers.
    query_cache_built = true;
    update_operator();
}

template <typename T>
void
TriangleMesh<T>::update_operator() const {
    // Update the query operator's raw pointers so they refer to the current
    // flattened query cache storage.
    _operator.vertices       = _query_vertices.empty() ? nullptr : atlas::raw_pointer_cast(_query_vertices.data());
    _operator.indices        = _query_indices.empty() ? nullptr : atlas::raw_pointer_cast(_query_indices.data());
    _operator.triangle_count = static_cast<int>(triangles.size());

    if (_bvh && bvh_built) {
        // If a valid BVH exists, export the BVH operator's raw acceleration data
        // into the mesh query operator.
        const auto bvh_op     = _bvh->make_geometry_operator();
        _operator.bvh_nodes   = bvh_op.bvh_nodes;
        _operator.bvh_indices = bvh_op.bvh_indices;
        _operator.bvh_tris    = bvh_op.bvh_tris;
        _operator.bvh_root    = bvh_op.bvh_root;
    } else {
        // Otherwise clear all BVH-related pointers so the operator knows that
        // no acceleration structure is available.
        _operator.bvh_nodes   = nullptr;
        _operator.bvh_indices = nullptr;
        _operator.bvh_tris    = nullptr;
        _operator.bvh_root    = -1;
    }
}

template <typename T>
GeometryOperator<T>
TriangleMesh<T>::make_geometry_operator() const {
    // Ensure flattened query data exists before exporting the generic operator.
    ensure_query_cache();

    // Wrap the mesh-specific query operator in the generic geometry-operator interface.
    return GeometryOperator<T>(_operator);
}

template <typename T>
bool
TriangleMesh<T>::load_from_obj(const std::string& filename, const bool verbose) {
    // Replace any currently stored mesh data with the newly loaded OBJ geometry.
    triangles.clear();
    query_cache_built = false;

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "";
    config.triangulate     = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config)) {
        // OBJ parsing failed.
        return false;
    }

    const tinyobj::attrib_t attrib             = reader.GetAttrib();
    const std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

    // A mesh with no vertex data is invalid for triangle generation.
    if (attrib.vertices.empty()) return false;

    const auto& v     = attrib.vertices;
    const auto vcount = v.size();

    for (const auto& shape : shapes) {
        const auto& idx = shape.mesh.indices;

        // `triangulate = true` guarantees that faces are represented as triangles,
        // so indices can be consumed in groups of 3.
        for (std::size_t f = 0; f + 2 < idx.size(); f += 3) {
            const int vi0 = idx[f + 0].vertex_index;
            const int vi1 = idx[f + 1].vertex_index;
            const int vi2 = idx[f + 2].vertex_index;

            // Skip faces with invalid OBJ vertex references.
            if (vi0 < 0 || vi1 < 0 || vi2 < 0) continue;

            const std::size_t o0 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi0);
            const std::size_t o1 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi1);
            const std::size_t o2 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi2);

            // Skip faces whose vertex indices would access past the raw OBJ vertex buffer.
            if (o0 + 2 >= vcount || o1 + 2 >= vcount || o2 + 2 >= vcount) continue;

            // Convert OBJ raw float vertex data into atlas vector types.
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

            // Materialize one triangle container from the face vertices.
            TriangleContainer4<T> tc;
            tc.a() = a;
            tc.b() = b;
            tc.c() = c;

            // Compute and store the face normal in the fourth slot.
            atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
            const T n2                  = n.length_squared();

            if (n2 > T(0)) {
                // Normalize non-degenerate face normals.
                n *= (T(1) / static_cast<T>(std::sqrt(n2)));
            } else {
                // Degenerate triangle fallback normal.
                n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
            }

            tc.d() = n;

            // Append the loaded triangle to mesh storage.
            triangles.push_back(tc);
        }
    }

    // Reject OBJ files that produced no usable triangles.
    if (triangles.empty()) return false;

    // Triangle storage changed, so any flattened cache must be rebuilt.
    query_cache_built = false;

    // Rebuild acceleration and query-side caches for the newly loaded mesh.
    ensure_bvh();
    build_bvh();
    ensure_query_cache();

    // The current implementation does not use the verbosity flag.
    (void)verbose;
    return true;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // The owning mesh may lazily build caches on the host, but device code must
    // use the already-materialized operator view.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Host code can build caches lazily; device code can only consume them.
    return _operator.closest_normal(p);
}

template <typename T>
T
TriangleMesh<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Host code can build caches lazily; device code can only consume them.
    return _operator.signed_distance(p);
}

template <typename T>
bool
TriangleMesh<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Host code can build caches lazily; device code can only consume them.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
TriangleMesh<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Host code can build caches lazily; device code can only consume them.

    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::centroid() const noexcept {
    // Host code can build caches lazily; device code can only consume them.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMesh<T>::bound() const noexcept {
    // Host code can build caches lazily; device code can only consume them.
    return _operator.bound();
}

template <typename T>
bool
TriangleMesh<T>::is_valid() const noexcept {
    // Host code can build caches lazily; device code can only consume them.
    return _operator.is_valid();
}

template <typename T>
GeometryType
TriangleMesh<T>::type() const noexcept {
    // Return the runtime geometry type tag for this concrete primitive.
    return GeometryType::TriangleMesh;
}

template <typename T>
TriangleMesh<T>
TriangleMesh<T>::Builder::build() const {
    // Validate staged builder input before constructing the mesh.
    validate();

    // Start from a default-constructed mesh and then populate its triangle storage.
    TriangleMesh<T> m {};

    m.triangles = _triangles;

    // Build all derived structures so the returned mesh is immediately query-ready.
    m.ensure_bvh();
    m.build_bvh();
    m.ensure_query_cache();

    return m;
}

template <typename T>
atlas::host_shared_ptr<TriangleMesh<T>>
TriangleMesh<T>::Builder::make_host_shared() const {
    // Build the mesh by value first, then move it into host-shared storage.
    auto m = build();
    return atlas::make_host_shared<TriangleMesh<T>>(std::move(m));
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(const HostBuffer<TriangleContainer4<T>>& ts) {
    // Copy triangle input into the builder's staged state.
    _triangles = ts;
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(HostBuffer<TriangleContainer4<T>>&& ts) {
    // Move triangle input into the builder's staged state.
    _triangles = std::move(ts);
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::load_from_obj(const std::string& filename, const bool verbose) {
    // Use a temporary mesh instance to reuse the mesh's OBJ loading path.
    TriangleMesh<T> tmp {};
    const bool ok = tmp.load_from_obj(filename, verbose);

    if (!ok) {
        throw std::runtime_error("TriangleMesh::Builder: failed to load OBJ.");
    }

    // Transfer only the loaded triangle data into the builder.
    // Derived caches will be rebuilt during `build()`.
    _triangles = std::move(tmp.triangles);
    return *this;
}

template <typename T>
void
TriangleMesh<T>::Builder::validate() const {
    // A triangle mesh must contain at least one triangle.
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
    // Compute the signed solid angle subtended by triangle (a, b, c) as seen from point `p`.
    //
    // This quantity is used by the winding-number test for inside/outside
    // classification of closed meshes.

    const atlas::math::Vector<T, 3> va = a - p;
    const atlas::math::Vector<T, 3> vb = b - p;
    const atlas::math::Vector<T, 3> vc = c - p;

    const T la = va.length();
    const T lb = vb.length();
    const T lc = vc.length();

    // If the query point is numerically on one of the triangle vertices,
    // avoid unstable division and return zero contribution.
    if (la <= std::numeric_limits<T>::epsilon() || lb <= std::numeric_limits<T>::epsilon()
        || lc <= std::numeric_limits<T>::epsilon()) {
        return T(0);
    }

    const T numerator   = va.dot(atlas::math::cross(vb, vc));
    const T denominator = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;

    // Standard robust solid-angle formula.
    return T(2) * std::atan2(numerator, denominator);
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::winding_number(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Sum signed solid angles over all triangles and normalize by 4*pi
    // to obtain the winding number at point `p`.
    T solid_angle_sum = T(0);

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        solid_angle_sum += solid_angle(p, a, b, c);
    }

    const T four_pi = T(4) * std::acos(T(-1));
    return solid_angle_sum / four_pi;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the mesh representation is invalid, return the query point unchanged.
    if (!is_valid()) return p;

    T best_d2                         = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_cp = p;

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Reconstruct a per-triangle normal for the temporary triangle operator.
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();

        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        // Keep the nearest closest-point candidate over all triangles.
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
    // If the mesh representation is invalid, return a stable fallback normal.
    if (!is_valid()) return atlas::math::Vector<T, 3>(T(0), T(0), T(1));

    T best_d2 = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_n(T(0), T(0), T(1));

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Reconstruct a normalized face normal for the temporary triangle query.
        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();
        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        // Use the normal of the triangle that owns the nearest closest point.
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
    // Invalid mesh representations cannot provide a meaningful signed distance.
    if (!is_valid()) return std::numeric_limits<T>::infinity();

    T best_d2 = std::numeric_limits<T>::max();

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();
        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
        }
    }

    const T dist = static_cast<T>(std::sqrt(best_d2));
    if (dist <= std::numeric_limits<T>::epsilon()) return T(0);

    // Use winding number for sign:
    // - |winding| > 0.5 -> inside a closed oriented mesh
    // - otherwise       -> outside
    const T winding = winding_number(p);
    return (std::abs(winding) > T(0.5)) ? -dist : dist;
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid mesh representations cannot classify containment.
    if (!is_valid()) return false;

    T best_d2 = std::numeric_limits<T>::max();

    TriangleGeometryOperator<T> tri {};

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
        const T n2                  = n.length_squared();
        if (n2 > T(0)) n *= (T(1) / static_cast<T>(std::sqrt(n2)));
        else
            n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));

        tri.a = &a;
        tri.b = &b;
        tri.c = &c;
        tri.n = &n;

        const atlas::math::Vector<T, 3> cp = tri.closest_point(p);
        const T d2                         = (cp - p).length_squared();

        if (d2 < best_d2) {
            best_d2 = d2;
        }
    }

    const T dist      = static_cast<T>(std::sqrt(best_d2));
    const T winding   = winding_number(p);
    const bool inside = std::abs(winding) > T(0.5);

    if (inside) {
        // Interior points are accepted immediately for non-negative tolerance.
        if (tolerance >= T(0)) return true;

        // Negative tolerance requires the point to be deep enough inside.
        return dist >= -tolerance;
    }

    // Exterior points cannot satisfy a negative tolerance band.
    if (tolerance < T(0)) return false;

    // For non-negative tolerance, allow a near-surface exterior shell.
    return dist <= tolerance;
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // A point is considered on the mesh surface when its absolute signed distance
    // falls within the specified tolerance band.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::centroid() const noexcept {
    // Invalid meshes use the origin as a safe fallback centroid.
    if (!is_valid()) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    atlas::math::Vector<T, 3> sum(T(0), T(0), T(0));

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        // Average the triangle centroids uniformly over all triangles.
        //
        // Note:
        // - this is an unweighted average over faces
        // - it is not an area-weighted surface centroid
        sum += (a + b + c) * (T(1) / T(3));
    }

    return sum * (T(1) / static_cast<T>(triangle_count));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMeshGeometryOperator<T>::bound() const noexcept {
    // Invalid meshes return a default-constructed bounding box.
    if (!is_valid()) return atlas::spatial::AxisAlignedBoundingBox<T>();

    // Initialize min/max bounds from the first referenced vertex.
    const int i0                 = indices[0];
    atlas::math::Vector<T, 3> mn = vertices[i0];
    atlas::math::Vector<T, 3> mx = vertices[i0];

    for (int t = 0; t < triangle_count; ++t) {
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

        // Expand the bounds by all three triangle vertices.
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
    // A valid query representation requires vertex/index storage and at least one triangle.
    if (!vertices || !indices) return false;
    if (triangle_count <= 0) return false;

    for (int t = 0; t < triangle_count; ++t) {
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

        const auto ab = vertices[j1] - vertices[j0];
        const auto ac = vertices[j2] - vertices[j0];
        const auto n  = atlas::math::cross(ab, ac);

        // Reject degenerate triangles whose area is effectively zero.
        if (!(n.length_squared() > static_cast<T>(atlas::eps))) {
            return false;
        }
    }

    return true;
}

template <typename T>
HitSurface<T>
TriangleMeshGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& r) const noexcept {
    // Start from the default "no hit" state.
    HitSurface<T> out {};

    // Ray tracing relies on BVH acceleration data.
    if (!bvh_nodes || !bvh_indices || !bvh_tris || bvh_root < 0) return out;

    T best_t = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_p {};
    atlas::math::Vector<T, 3> best_n {};
    bool found = false;

    // Use a fixed-size explicit traversal stack for iterative BVH traversal.
    int stack[64];
    int sp      = 0;
    stack[sp++] = bvh_root;

    while (sp) {
        const int ni                         = stack[--sp];
        const atlas::spatial::BVHNode<T>& nd = bvh_nodes[ni];

        // First test the ray against the node's bounding box.
        const auto hit = nd.bounds.trace(r);
        if (!hit.is_intersecting || hit.enter > best_t) continue;

        if (nd.is_leaf) {
            // Leaf nodes contain a contiguous range of triangle references.
            TriangleGeometryOperator<T> tri_op {};

            ATLAS_UNROLL
            for (int k = 0; k < nd.count; ++k) {
                const int pid                    = bvh_indices[nd.start + k];
                const TriangleContainer4<T>& tri = bvh_tris[pid];

                // Bind the temporary triangle operator directly to the leaf triangle data.
                tri_op.a = &tri.a();
                tri_op.b = &tri.b();
                tri_op.c = &tri.c();
                tri_op.n = &tri.d();

                const HitSurface<T> h = tri_op.trace(r);
                if (h.is_intersecting && h.distance < best_t) {
                    // Keep only the nearest triangle hit found so far.
                    best_t = h.distance;
                    best_p = h.point;
                    best_n = h.normal;
                    found  = true;
                }
            }
        } else {
            // Push child nodes onto the traversal stack.
            //
            // The code uses a defensive cap at the end of the fixed array to avoid
            // writing past the stack bound.
            if (sp < 63) stack[sp++] = nd.left;
            else
                stack[63] = nd.left;

            if (sp < 63) stack[sp++] = nd.right;
            else
                stack[63] = nd.right;
        }
    }

    if (found) {
        // Populate the final nearest-hit record.
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
    // Function-call convenience wrapper around the explicit trace routine.
    return trace(ray);
}

} // namespace atlas::geometry
