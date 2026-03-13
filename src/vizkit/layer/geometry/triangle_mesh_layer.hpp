#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
TriangleMeshLayer<T>::TriangleMeshLayer(const atlas::UnitHostPtr<T>& unit)
    // Meshes are rendered as a flat triangle list, so GL_TRIANGLES matches the
    // expanded vertex stream produced in build_geometry().
    : GeometryLayer<T>(GL_TRIANGLES, unit) { }

template <typename T>
typename TriangleMeshLayer<T>::Builder
TriangleMeshLayer<T>::builder() {
    return Builder {};
}

template <typename T>
void
TriangleMeshLayer<T>::build_geometry(std::vector<Vector3<T>>& pos) {

    // Expand indexed mesh data into a non-indexed triangle stream.
    //
    // The base GeometryLayer uses glDrawArrays, not glDrawElements, so indices
    // are resolved on the CPU by duplicating referenced vertices.
    pos.clear();
    if (!this->_unit) return;

    auto& q = this->_unit->query_operator();
    if (q.type != atlas::geometry::GeometryType::TriangleMesh) return;

    auto& m = q.triangle_mesh;
    if (!m.vertices || !m.indices) return;

    // Each triangle contributes exactly 3 vertex references:
    //   (i0, i1, i2)
    //
    // The expansion below converts the mesh from indexed representation
    //   vertices + indices
    // into the explicit representation required by glDrawArrays:
    //   [v(i0), v(i1), v(i2), v(i3), v(i4), v(i5), ...]
    //
    // This increases memory traffic but keeps the render path extremely simple.
    for (int i = 0; i < m.triangle_count; i++) {
        int i0 = m.indices[3 * i];
        int i1 = m.indices[3 * i + 1];
        int i2 = m.indices[3 * i + 2];

        // The winding order coming from the index buffer is preserved exactly.
        // That matters for rasterization if back-face culling or normal
        // reconstruction depends on consistent orientation.
        pos.push_back(m.vertices[i0]);
        pos.push_back(m.vertices[i1]);
        pos.push_back(m.vertices[i2]);
    }
}

template <typename T>
typename TriangleMeshLayer<T>::Builder&
TriangleMeshLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& u) {
    _unit = u;
    return *this;
}

template <typename T>
void
TriangleMeshLayer<T>::Builder::validate() const {
    // Minimal validation mirrors the current implementation style of the other
    // mesh-like layers: require a unit and leave deeper mesh integrity checks
    // to the query data producer.
    if (!_unit) throw std::runtime_error("TriangleMeshLayer unit null");
}

template <typename T>
TriangleMeshLayer<T>
TriangleMeshLayer<T>::Builder::build() const {
    validate();
    return TriangleMeshLayer(_unit);
}

template <typename T>
std::shared_ptr<TriangleMeshLayer<T>>
TriangleMeshLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<TriangleMeshLayer<T>>(_unit);
}

}

#endif
