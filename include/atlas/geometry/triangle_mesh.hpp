#pragma once
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/math/math.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>
#include <cmath>
#include <limits>
#include <tiny_obj_loader.h>

namespace atlas::geometry {
template <typename T>
TriangleMesh<T>::TriangleMesh(const HostBuffer<Triangle<T>>& triangles_) noexcept
    : triangles(triangles_) {
    _bvh = atlas::make_host_shared<SAHBVH<T>>();
    build_bvh();
}

template <typename T>
TriangleMesh<T>::TriangleMesh(HostBuffer<Triangle<T>>&& triangles_) noexcept
    : triangles(std::move(triangles_)) {
    _bvh = atlas::make_host_shared<SAHBVH<T>>();
    build_bvh();
}

template <typename T>
void
TriangleMesh<T>::set_bvh(const BVHHostPtr<T>& bvh) noexcept {
    _bvh = bvh;
}

template <typename T>
void
TriangleMesh<T>::set_triangles(const HostBuffer<Triangle<T>>& triangles_) {
    triangles = triangles_;
}

template <typename T>
Vector3<T>
TriangleMesh<T>::closest_point(const Vector3<T>& p) const {
    if (triangles.empty()) {
        return p;
    }
    T min_dist2           = std::numeric_limits<T>::max();
    Vector3<T> best_point = p;
    for (const auto& tri : triangles) {
        const Vector3<T> cp = tri.closest_point(p);
        const T dist2       = math::length_squared(cp - p);
        if (dist2 < min_dist2) {
            min_dist2  = dist2;
            best_point = cp;
        }
    }
    return best_point;
}

template <typename T>
Vector3<T>
TriangleMesh<T>::closest_normal(const Vector3<T>& p) const {
    if (triangles.empty()) {
        return Vector3<T>(T(0), T(0), T(1));
    }
    T min_dist2                 = std::numeric_limits<T>::max();
    const Triangle<T>* best_tri = nullptr;
    for (const auto& tri : triangles) {
        const Vector3<T> cp = tri.closest_point(p);
        const T dist2       = math::length_squared(cp - p);
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            best_tri  = &tri;
        }
    }
    if (best_tri) {
        return best_tri->closest_normal(p);
    }
    return Vector3<T>(T(0), T(0), T(1));
}

template <typename T>
T
TriangleMesh<T>::closest_distance(const Vector3<T>& p) const {
    const Vector3<T> cp = closest_point(p);
    return math::length(cp - p);
}

template <typename T>
T
TriangleMesh<T>::signed_distance(const Vector3<T>& p) const {
    if (triangles.empty()) {
        return std::numeric_limits<T>::max();
    }
    const T unsigned_dist = closest_distance(p);
    const T w             = fast_winding_number(p);
    const bool inside     = std::abs(w) > T(0.5);
    return inside ? -unsigned_dist : unsigned_dist;
}

template <typename T>
bool
TriangleMesh<T>::is_inside(const Vector3<T>& p) const {
    if (triangles.empty()) {
        return false;
    }
    const T w = fast_winding_number(p);
    return std::abs(w) > T(0.5);
}

template <typename T>
spatial::AxisAlignedBoundingBox<T>
TriangleMesh<T>::bound() const {
    if (triangles.empty()) {
        return spatial::AxisAlignedBoundingBox<T>();
    }
    spatial::AxisAlignedBoundingBox<T> box = triangles.front().bound();
    for (std::size_t i = 1; i < triangles.size(); ++i) {
        const auto tri_box = triangles[i].bound();
        box.lower_corner   = math::cmin(box.lower_corner, tri_box.lower_corner);
        box.upper_corner   = math::cmax(box.upper_corner, tri_box.upper_corner);
    }
    return box;
}

template <typename T>
bool
TriangleMesh<T>::intersects(const spatial::Ray<T>& ray) const {
    for (const auto& tri : triangles) {
        if (tri.intersects(ray)) {
            return true;
        }
    }
    return false;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE

bool
TriangleMesh<T>::load_from_obj(const std::string& filename, bool verbose) {
    triangles.clear();
    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = "";
    config.triangulate     = true;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filename, config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader Error: " << reader.Error() << "\n";
        } else {
            std::cerr << "TinyObjReader: Failed to read file: " << filename << "\n";
        }
        return false;
    }
    if (!reader.Warning().empty() && verbose) {
        std::cerr << "TinyObjReader Warning: " << reader.Warning() << "\n";
    }
    tinyobj::attrib_t attrib                   = reader.GetAttrib();
    std::vector<tinyobj::shape_t> shapes       = reader.GetShapes();
    std::vector<tinyobj::material_t> materials = reader.GetMaterials();
    if (attrib.vertices.empty()) {
        if (verbose) {
            std::cerr << "No vertices in OBJ file: " << filename << "\n";
        }
        return false;
    }
    for (const auto& shape : shapes) {
        const auto& mesh = shape.mesh;
        if (mesh.indices.size() % 3 != 0 && verbose) {
            std::cerr << "Warning: mesh.indices.size() is not multiple of 3 in shape \""
                << shape.name << "\"\n";
        }
        for (size_t f = 0; f + 2 < mesh.indices.size(); f += 3) {
            const tinyobj::index_t idx0 = mesh.indices[f + 0];
            const tinyobj::index_t idx1 = mesh.indices[f + 1];
            const tinyobj::index_t idx2 = mesh.indices[f + 2];
            const int vi0               = idx0.vertex_index;
            const int vi1               = idx1.vertex_index;
            const int vi2               = idx2.vertex_index;
            if (vi0 < 0 || vi1 < 0 || vi2 < 0) {
                continue;
            }
            const size_t v0_offset = static_cast<size_t>(3 * vi0);
            const size_t v1_offset = static_cast<size_t>(3 * vi1);
            const size_t v2_offset = static_cast<size_t>(3 * vi2);
            if (v0_offset + 2 >= attrib.vertices.size() || v1_offset + 2 >= attrib.vertices.size() || v2_offset + 2
                >= attrib.vertices.size()) {
                if (verbose) {
                    std::cerr << "Invalid vertex offset in OBJ file\n";
                }
                continue;
            }
            Vector3<T> a(
                static_cast<T>(attrib.vertices[v0_offset + 0]),
                static_cast<T>(attrib.vertices[v0_offset + 1]),
                static_cast<T>(attrib.vertices[v0_offset + 2]));
            Vector3<T> b(
                static_cast<T>(attrib.vertices[v1_offset + 0]),
                static_cast<T>(attrib.vertices[v1_offset + 1]),
                static_cast<T>(attrib.vertices[v1_offset + 2]));
            Vector3<T> c(
                static_cast<T>(attrib.vertices[v2_offset + 0]),
                static_cast<T>(attrib.vertices[v2_offset + 1]),
                static_cast<T>(attrib.vertices[v2_offset + 2]));
            triangles.push_back(Triangle<T>(a, b, c));
        }
    }
    if (verbose) {
        std::cout << "Loaded " << triangles.size()
            << " triangles from OBJ: " << filename << "\n";
    }
    const bool ok = !triangles.empty();
    if (ok) {
        build_bvh();
    }
    return ok;
}

template <typename T>
std::vector<Triangle<T>>
TriangleMesh<T>::to_vector() const {
    return std::vector<Triangle<T>>(triangles.begin(), triangles.end());
}

template <typename T>
BvhTraceOperator<T>
TriangleMesh<T>::make_trace_operator() const {
    if (_bvh && bvh_built) {
        return _bvh->make_trace_operator();
    } else {
        return BvhTraceOperator<T>();
    }
}

template <typename T>
void
TriangleMesh<T>::build_bvh() {
    if (_bvh) {
        _bvh->build(triangles);
        bvh_built = true;
    }
}

template <typename T>
void
TriangleMesh<T>::invalidate_bvh() noexcept {
    bvh_built = false;
}

template <typename T>
T
TriangleMesh<T>::fast_winding_number(const Vector3<T>& p) const {
    if (triangles.empty()) {
        return T(0);
    }
    const T pi          = T(3.1415926535897932384626433832795);
    const T fourPi      = T(4) * pi;
    T total_solid_angle = T(0);
    for (const auto& tri : triangles) {
        const Vector3<T> r0 = tri.a - p;
        const Vector3<T> r1 = tri.b - p;
        const Vector3<T> r2 = tri.c - p;
        const T l0          = math::length(r0);
        const T l1          = math::length(r1);
        const T l2          = math::length(r2);
        if (l0 == T(0) || l1 == T(0) || l2 == T(0)) {
            continue;
        }
        const T dot01            = math::dot(r0, r1);
        const T dot12            = math::dot(r1, r2);
        const T dot20            = math::dot(r2, r0);
        const Vector3<T> cross01 = math::cross(r0, r1);
        const T numer            = math::dot(cross01, r2);
        const T denom            = l0 * l1 * l2 + dot01 * l2 + dot12 * l0 + dot20 * l1;
        const T omega            = T(2) * std::atan2(numer, denom);
        total_solid_angle        += omega;
    }
    return total_solid_angle / fourPi;
}
}