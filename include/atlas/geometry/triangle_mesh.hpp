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

    ensure_bvh();
    build_bvh();
    ensure_query_cache();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(HostBuffer<TriangleContainer4<T>>&& triangles_) noexcept
    : triangles(std::move(triangles_)) {

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
    update_operator();

    other.bvh_built = false;
    other.query_cache_built = false;
    other.update_operator();
}

template <typename T>
TriangleMesh<T>&
TriangleMesh<T>::operator=(const TriangleMesh& other) {
    if (this == &other) return *this;

    triangles = other.triangles;
    _bvh = other._bvh;
    _query_vertices = other._query_vertices;
    _query_indices = other._query_indices;
    bvh_built = other.bvh_built;
    query_cache_built = other.query_cache_built;
    update_operator();
    return *this;
}

template <typename T>
TriangleMesh<T>&
TriangleMesh<T>::operator=(TriangleMesh&& other) noexcept {
    if (this == &other) return *this;

    triangles = std::move(other.triangles);
    _bvh = std::move(other._bvh);
    _query_vertices = std::move(other._query_vertices);
    _query_indices = std::move(other._query_indices);
    bvh_built = other.bvh_built;
    query_cache_built = other.query_cache_built;
    update_operator();

    other.bvh_built = false;
    other.query_cache_built = false;
    other.update_operator();
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder
TriangleMesh<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
void
TriangleMesh<T>::set_triangles(const HostBuffer<TriangleContainer4<T>>& triangles_) {

    triangles         = triangles_;
    query_cache_built = false;
    this->invalidate_validity_cache();
    ensure_bvh();
    build_bvh();
    ensure_query_cache();
}

template <typename T>
void
TriangleMesh<T>::ensure_bvh() noexcept {

    if (!_bvh) _bvh = atlas::make_host_shared<SAHBVH<T>>();
}

template <typename T>
void
TriangleMesh<T>::build_bvh() {

    if (!_bvh) return;

    if (triangles.empty()) {
        bvh_built = false;
        return;
    }

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
        const auto& tri        = triangles[t];
        const std::size_t base = t * 3;

        _query_vertices[base + 0] = tri.a();
        _query_vertices[base + 1] = tri.b();
        _query_vertices[base + 2] = tri.c();

        _query_indices[base + 0] = static_cast<int>(base + 0);
        _query_indices[base + 1] = static_cast<int>(base + 1);
        _query_indices[base + 2] = static_cast<int>(base + 2);
    }

    query_cache_built = true;
    update_operator();
}

template <typename T>
void
TriangleMesh<T>::update_operator() const {
    _operator.vertices       = _query_vertices.empty() ? nullptr : atlas::raw_pointer_cast(_query_vertices.data());
    _operator.indices        = _query_indices.empty() ? nullptr : atlas::raw_pointer_cast(_query_indices.data());
    _operator.triangle_count = static_cast<int>(triangles.size());
    this->invalidate_validity_cache();

    if (_bvh && bvh_built) {
        const auto bvh_op    = _bvh->make_geometry_operator();
        _operator.bvh_nodes   = bvh_op.bvh_nodes;
        _operator.bvh_indices = bvh_op.bvh_indices;
        _operator.bvh_tris    = bvh_op.bvh_tris;
        _operator.bvh_root    = bvh_op.bvh_root;
    } else {
        _operator.bvh_nodes   = nullptr;
        _operator.bvh_indices = nullptr;
        _operator.bvh_tris    = nullptr;
        _operator.bvh_root    = -1;
    }
}

template <typename T>
GeometryOperator<T>
TriangleMesh<T>::make_geometry_operator() const {

    ensure_query_cache();

    return GeometryOperator<T>(_operator);
}

template <typename T>
bool
TriangleMesh<T>::load_from_obj(const std::string& filename, const bool verbose) {

    triangles.clear();
    query_cache_built = false;
    this->invalidate_validity_cache();

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "";
    config.triangulate     = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config)) {

        return false;
    }

    const tinyobj::attrib_t attrib             = reader.GetAttrib();
    const std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

    if (attrib.vertices.empty()) return false;

    const auto& v     = attrib.vertices;
    const auto vcount = v.size();

    for (const auto& shape : shapes) {
        const auto& idx = shape.mesh.indices;

        for (std::size_t f = 0; f + 2 < idx.size(); f += 3) {
            const int vi0 = idx[f + 0].vertex_index;
            const int vi1 = idx[f + 1].vertex_index;
            const int vi2 = idx[f + 2].vertex_index;

            if (vi0 < 0 || vi1 < 0 || vi2 < 0) continue;

            const std::size_t o0 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi0);
            const std::size_t o1 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi1);
            const std::size_t o2 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi2);

            if (o0 + 2 >= vcount || o1 + 2 >= vcount || o2 + 2 >= vcount) continue;

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
            tc.a() = a;
            tc.b() = b;
            tc.c() = c;

            atlas::math::Vector<T, 3> n = atlas::math::cross(b - a, c - a);
            const T n2                  = n.length_squared();

            if (n2 > T(0)) {
                n *= (T(1) / static_cast<T>(std::sqrt(n2)));
            } else {

                n = atlas::math::Vector<T, 3>(T(0), T(0), T(1));
            }

            tc.d() = n;

            triangles.push_back(tc);
        }
    }

    if (triangles.empty()) return false;

    query_cache_built = false;

    ensure_bvh();
    build_bvh();
    ensure_query_cache();

    (void)verbose;
    return true;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    ensure_query_cache();
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    ensure_query_cache();
    return _operator.closest_normal(p);
}

template <typename T>
T
TriangleMesh<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    ensure_query_cache();
    return _operator.signed_distance(p);
}

template <typename T>
bool
TriangleMesh<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    ensure_query_cache();
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
TriangleMesh<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    ensure_query_cache();
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMesh<T>::centroid() const noexcept {
    ensure_query_cache();
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMesh<T>::bound() const noexcept {
    ensure_query_cache();
    return _operator.bound();
}

template <typename T>
bool
TriangleMesh<T>::is_valid() const noexcept {
    ensure_query_cache();
    return this->cached_is_valid([this]() noexcept {
        return _operator.is_valid();
    });
}

template <typename T>
GeometryType
TriangleMesh<T>::type() const noexcept {

    return GeometryType::TriangleMesh;
}

template <typename T>
TriangleMesh<T>
TriangleMesh<T>::Builder::build() const {

    validate();

    TriangleMesh<T> m {};

    m.triangles = _triangles;

    m.ensure_bvh();
    m.build_bvh();
    m.ensure_query_cache();

    return m;
}

template <typename T>
atlas::host_shared_ptr<TriangleMesh<T>>
TriangleMesh<T>::Builder::make_host_shared() const {

    auto m = build();
    return atlas::make_host_shared<TriangleMesh<T>>(std::move(m));
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(const HostBuffer<TriangleContainer4<T>>& ts) {

    _triangles = ts;
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::with_triangles(HostBuffer<TriangleContainer4<T>>&& ts) {

    _triangles = std::move(ts);
    return *this;
}

template <typename T>
typename TriangleMesh<T>::Builder&
TriangleMesh<T>::Builder::load_from_obj(const std::string& filename, const bool verbose) {

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

    if (_triangles.empty()) {
        atlas::logger::error()
            << "TriangleMesh::Builder validation failed: no triangles provided.";
        throw std::runtime_error("TriangleMesh::Builder validation failed: no triangles provided.");
    }
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::solid_angle(const atlas::math::Vector<T, 3>& p,
                                             const atlas::math::Vector<T, 3>& a,
                                             const atlas::math::Vector<T, 3>& b,
                                             const atlas::math::Vector<T, 3>& c) const noexcept {
    const atlas::math::Vector<T, 3> va = a - p;
    const atlas::math::Vector<T, 3> vb = b - p;
    const atlas::math::Vector<T, 3> vc = c - p;

    const T la = va.length();
    const T lb = vb.length();
    const T lc = vc.length();

    if (la <= std::numeric_limits<T>::epsilon() || lb <= std::numeric_limits<T>::epsilon()
        || lc <= std::numeric_limits<T>::epsilon()) {
        return T(0);
    }

    const T numerator   = va.dot(atlas::math::cross(vb, vc));
    const T denominator = la * lb * lc + va.dot(vb) * lc + vb.dot(vc) * la + vc.dot(va) * lb;

    return T(2) * std::atan2(numerator, denominator);
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::winding_number(const atlas::math::Vector<T, 3>& p) const noexcept {
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
            best_cp = cp;
        }
    }

    return best_cp;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
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
            best_n  = tri.closest_normal(p);
        }
    }

    return best_n;
}

template <typename T>
T
TriangleMeshGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
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

    const T winding = winding_number(p);
    return (std::abs(winding) > T(0.5)) ? -dist : dist;
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
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
        if (tolerance >= T(0)) return true;
        return dist >= -tolerance;
    }

    if (tolerance < T(0)) return false;
    return dist <= tolerance;
}

template <typename T>
bool
TriangleMeshGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
TriangleMeshGeometryOperator<T>::centroid() const noexcept {
    if (!is_valid()) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    atlas::math::Vector<T, 3> sum(T(0), T(0), T(0));

    for (int t = 0; t < triangle_count; ++t) {
        const int i0 = indices[3 * t + 0];
        const int i1 = indices[3 * t + 1];
        const int i2 = indices[3 * t + 2];

        const atlas::math::Vector<T, 3>& a = vertices[i0];
        const atlas::math::Vector<T, 3>& b = vertices[i1];
        const atlas::math::Vector<T, 3>& c = vertices[i2];

        sum += (a + b + c) * (T(1) / T(3));
    }

    return sum * (T(1) / static_cast<T>(triangle_count));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
TriangleMeshGeometryOperator<T>::bound() const noexcept {
    if (!is_valid()) return atlas::spatial::AxisAlignedBoundingBox<T>();

    const int i0                 = indices[0];
    atlas::math::Vector<T, 3> mn = vertices[i0];
    atlas::math::Vector<T, 3> mx = vertices[i0];

    for (int t = 0; t < triangle_count; ++t) {
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

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
    if (!vertices || !indices) return false;
    if (triangle_count <= 0) return false;

    for (int t = 0; t < triangle_count; ++t) {
        const int j0 = indices[3 * t + 0];
        const int j1 = indices[3 * t + 1];
        const int j2 = indices[3 * t + 2];

        const auto ab = vertices[j1] - vertices[j0];
        const auto ac = vertices[j2] - vertices[j0];
        const auto n = atlas::math::cross(ab, ac);

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
    if (!bvh_nodes || !bvh_indices || !bvh_tris || bvh_root < 0) return out;

    T best_t = std::numeric_limits<T>::max();
    atlas::math::Vector<T, 3> best_p {};
    atlas::math::Vector<T, 3> best_n {};
    bool found = false;

    int stack[64];
    int sp      = 0;
    stack[sp++] = bvh_root;

    while (sp) {
        const int ni                         = stack[--sp];
        const atlas::spatial::BVHNode<T>& nd = bvh_nodes[ni];
        const auto hit                       = nd.bounds.trace(r);
        if (!hit.is_intersecting || hit.enter > best_t) continue;

        if (nd.is_leaf) {
            TriangleGeometryOperator<T> tri_op {};
            ATLAS_UNROLL
            for (int k = 0; k < nd.count; ++k) {
                const int pid                    = bvh_indices[nd.start + k];
                const TriangleContainer4<T>& tri = bvh_tris[pid];
                tri_op.a                         = &tri.a();
                tri_op.b                         = &tri.b();
                tri_op.c                         = &tri.c();
                tri_op.n                         = &tri.d();

                const HitSurface<T> h = tri_op.trace(r);
                if (h.is_intersecting && h.distance < best_t) {
                    best_t = h.distance;
                    best_p = h.point;
                    best_n = h.normal;
                    found  = true;
                }
            }
        } else {
            if (sp < 63) stack[sp++] = nd.left;
            else
                stack[63] = nd.left;
            if (sp < 63) stack[sp++] = nd.right;
            else
                stack[63] = nd.right;
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

template <typename T>
HitSurface<T>
TriangleMeshGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    return trace(ray);
}

}
