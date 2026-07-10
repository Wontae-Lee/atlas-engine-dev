#include <atlas/geometry/geometry.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <cstddef>
#include <stdexcept>
#include <tiny_obj_loader.h>
#include <utility>
#include <vector>

namespace atlas {

TriangleMesh::TriangleMesh(const HostBuffer<TriangleContainer4>& triangles_) noexcept
    : triangles(triangles_) {

    ensure_bvh();

    build_bvh();

    ensure_query_cache();
}

TriangleMesh::TriangleMesh(HostBuffer<TriangleContainer4>&& triangles_) noexcept
    : triangles(std::move(triangles_)) {

    ensure_bvh();

    build_bvh();

    ensure_query_cache();
}

TriangleMesh::TriangleMesh(const TriangleMesh& other)
    : triangles(other.triangles)
    , _bvh(other._bvh)
    , _query_vertices(other._query_vertices)
    , _query_indices(other._query_indices)
    , bvh_built(other.bvh_built)
    , query_cache_built(other.query_cache_built) {

    update_view();
}

TriangleMesh::TriangleMesh(TriangleMesh&& other) noexcept
    : triangles(std::move(other.triangles))
    , _bvh(std::move(other._bvh))
    , _query_vertices(std::move(other._query_vertices))
    , _query_indices(std::move(other._query_indices))
    , bvh_built(other.bvh_built)
    , query_cache_built(other.query_cache_built) {

    update_view();

    other.bvh_built         = false;
    other.query_cache_built = false;

    other.update_view();
}

TriangleMesh&
TriangleMesh::operator=(const TriangleMesh& other) {

    if (this == &other) {
        return *this;
    }

    triangles         = other.triangles;
    _bvh              = other._bvh;
    _query_vertices   = other._query_vertices;
    _query_indices    = other._query_indices;
    bvh_built         = other.bvh_built;
    query_cache_built = other.query_cache_built;

    update_view();

    return *this;
}

TriangleMesh&
TriangleMesh::operator=(TriangleMesh&& other) noexcept {

    if (this == &other) {
        return *this;
    }

    triangles         = std::move(other.triangles);
    _bvh              = std::move(other._bvh);
    _query_vertices   = std::move(other._query_vertices);
    _query_indices    = std::move(other._query_indices);
    bvh_built         = other.bvh_built;
    query_cache_built = other.query_cache_built;

    update_view();

    other.bvh_built         = false;
    other.query_cache_built = false;
    other.update_view();

    return *this;
}

TriangleMesh::Builder
TriangleMesh::builder() noexcept {
    return Builder {};
}

void
TriangleMesh::set_triangles(const HostBuffer<TriangleContainer4>& triangles_) {

    triangles = triangles_;

    query_cache_built = false;

    ensure_bvh();
    build_bvh();
    ensure_query_cache();
}

void
TriangleMesh::ensure_bvh() noexcept {

    if (!_bvh) {
        _bvh = atlas::make_host_shared<SAHBVH>();
    }
}

void
TriangleMesh::build_bvh() {

    if (!_bvh) {
        return;
    }

    if (triangles.empty()) {
        bvh_built = false;
        return;
    }

    _bvh->build(triangles);
    bvh_built = true;
}

void
TriangleMesh::ensure_query_cache() const {

    if (query_cache_built) {
        return;
    }

    rebuild_query_cache();
}

void
TriangleMesh::rebuild_query_cache() const {
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

    update_view();
}

void
TriangleMesh::update_view() const {

    _view.vertices       = _query_vertices.empty() ? nullptr : atlas::raw_pointer_cast(_query_vertices.data());
    _view.indices        = _query_indices.empty() ? nullptr : atlas::raw_pointer_cast(_query_indices.data());
    _view.triangle_count = static_cast<int>(triangles.size());

    if (_bvh && bvh_built) {

        const auto bvh_view     = _bvh->view();
        _view.bvh_nodes   = bvh_view.bvh_nodes;
        _view.bvh_indices = bvh_view.bvh_indices;
        _view.bvh_tris    = bvh_view.bvh_tris;
        _view.bvh_root    = bvh_view.bvh_root;
    } else {

        _view.bvh_nodes   = nullptr;
        _view.bvh_indices = nullptr;
        _view.bvh_tris    = nullptr;
        _view.bvh_root    = -1;
    }
}

Geometry
TriangleMesh::make_device_geometry_view() const {

    ensure_query_cache();

    return Geometry(_view);
}

bool
TriangleMesh::load_from_obj(const std::string& filename, const bool verbose) {

    triangles.clear();
    query_cache_built = false;

    tinyobj::ObjReaderConfig config;

    config.mtl_search_path = "";
    config.triangulate     = true;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, config)) {
        return false;
    }

    const tinyobj::attrib_t attrib             = reader.GetAttrib();
    const std::vector<tinyobj::shape_t> shapes = reader.GetShapes();

    if (attrib.vertices.empty()) {
        return false;
    }

    const auto& v     = attrib.vertices;
    const auto vcount = v.size();

    for (const auto& shape : shapes) {
        const auto& idx = shape.mesh.indices;

        for (std::size_t f = 0; f + 2 < idx.size(); f += 3) {
            const int vi0 = idx[f + 0].vertex_index;
            const int vi1 = idx[f + 1].vertex_index;
            const int vi2 = idx[f + 2].vertex_index;

            if (vi0 < 0 || vi1 < 0 || vi2 < 0) {
                continue;
            }

            const std::size_t o0 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi0);
            const std::size_t o1 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi1);
            const std::size_t o2 = static_cast<std::size_t>(3) * static_cast<std::size_t>(vi2);

            if (o0 + 2 >= vcount || o1 + 2 >= vcount || o2 + 2 >= vcount) {
                continue;
            }

            const Float3 a(
                static_cast<float>(v[o0 + 0]),
                static_cast<float>(v[o0 + 1]),
                static_cast<float>(v[o0 + 2]));

            const Float3 b(
                static_cast<float>(v[o1 + 0]),
                static_cast<float>(v[o1 + 1]),
                static_cast<float>(v[o1 + 2]));

            const Float3 c(
                static_cast<float>(v[o2 + 0]),
                static_cast<float>(v[o2 + 1]),
                static_cast<float>(v[o2 + 2]));

            TriangleContainer4 tc;

            tc.a() = a;
            tc.b() = b;
            tc.c() = c;

            tc.d() = atlas::normalized_or(
                atlas::cross(b - a, c - a),
                Float3(0.0f, 0.0f, 1.0f));

            triangles.push_back(tc);
        }
    }

    if (triangles.empty()) {
        return false;
    }

    query_cache_built = false;
    ensure_bvh();
    build_bvh();
    ensure_query_cache();

    (void)verbose;

    return true;
}

Float3
TriangleMesh::closest_point(const Float3& p) const noexcept {
    return _view.closest_point(p);
}

Float3
TriangleMesh::closest_normal(const Float3& p) const noexcept {
    return _view.closest_normal(p);
}

float
TriangleMesh::signed_distance(const Float3& p) const noexcept {
    return _view.signed_distance(p);
}

bool
TriangleMesh::is_inside(const Float3& p, const float tolerance) const noexcept {
    return _view.is_inside(p, tolerance);
}

bool
TriangleMesh::is_on_surface(const Float3& p, const float tolerance) const noexcept {
    return _view.is_on_surface(p, tolerance);
}

Float3
TriangleMesh::centroid() const noexcept {
    return _view.centroid();
}

AABB
TriangleMesh::bound() const noexcept {
    return _view.bound();
}

bool
TriangleMesh::is_valid() const noexcept {
    return _view.is_valid();
}

TriangleMesh
TriangleMesh::Builder::build() const {

    validate();

    TriangleMesh m {};
    m.triangles = _triangles;

    m.ensure_bvh();
    m.build_bvh();
    m.ensure_query_cache();

    return m;
}

atlas::host_shared_ptr<TriangleMesh>
TriangleMesh::Builder::make_host_shared() const {

    auto m = build();
    return atlas::make_host_shared<TriangleMesh>(std::move(m));
}

TriangleMesh::Builder&
TriangleMesh::Builder::with_triangles(const HostBuffer<TriangleContainer4>& ts) {

    _triangles = ts;
    return *this;
}

TriangleMesh::Builder&
TriangleMesh::Builder::with_triangles(HostBuffer<TriangleContainer4>&& ts) {

    _triangles = std::move(ts);
    return *this;
}

TriangleMesh::Builder&
TriangleMesh::Builder::load_from_obj(const std::string& filename, const bool verbose) {
    TriangleMesh tmp {};

    const bool ok = tmp.load_from_obj(filename, verbose);

    if (!ok) {
        throw std::runtime_error("TriangleMesh::Builder: failed to load OBJ.");
    }

    _triangles = std::move(tmp.triangles);

    return *this;
}

void
TriangleMesh::Builder::validate() const {

    if (_triangles.empty()) {
        throw std::runtime_error("TriangleMesh::Builder validation failed: no triangles provided.");
    }
}

}
