#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
TriangleMeshLayer<T>::TriangleMeshLayer(const atlas::UnitHostPtr<T>& unit)

    : GeometryLayer<T>(GL_TRIANGLES, unit) { }

template <typename T>
typename TriangleMeshLayer<T>::Builder
TriangleMeshLayer<T>::builder() {
    return Builder {};
}

template <typename T>
void
TriangleMeshLayer<T>::build_geometry(std::vector<Vector3<T>>& pos) {

    pos.clear();
    if (!this->_unit) return;

    const auto& q = this->_unit->geometry_operator();
    if (q.type != atlas::geometry::GeometryType::TriangleMesh) return;

    auto& m = q.triangle_mesh;
    if (!m.vertices || !m.indices) return;

    for (int i = 0; i < m.triangle_count; i++) {
        int i0 = m.indices[3 * i];
        int i1 = m.indices[3 * i + 1];
        int i2 = m.indices[3 * i + 2];

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
