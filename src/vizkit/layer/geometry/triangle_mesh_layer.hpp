#pragma once

#include <stdexcept>

namespace atlas::vizkit {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
TriangleMeshLayer<T>::TriangleMeshLayer(std::vector<Vector3<T>> vertices)
    : GeometryLayer<T>(GL_TRIANGLES)
    , _vertices(std::move(vertices)) {

    if (_vertices.size() % 3 != 0) {
        throw std::runtime_error("TriangleMeshLayer: vertex count must be a multiple of 3.");
    }
}

template <typename T>
void
TriangleMeshLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    if (_vertices.empty()) {
        return;
    }


    const std::size_t tri_vertex_count = (_vertices.size() / 3) * 3;

    positions.reserve(tri_vertex_count);
    for (std::size_t i = 0; i < tri_vertex_count; ++i) {
        positions.push_back(_vertices[i]);
    }
}

}