#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
TriangleMeshLayer<T>::TriangleMeshLayer(const atlas::UnitHostPtr<T>& unit)
    : GeometryLayer<T>(GL_TRIANGLES, unit) {
    // Construct a triangle-mesh visualization layer.
    //
    // Rendering mode:
    // - GL_TRIANGLES
    //
    // Geometry policy:
    // - every mesh triangle is expanded into 3 explicit vertex positions
    // - indexed mesh connectivity is converted into a flat triangle list
    //
    // The bound unit is forwarded to GeometryLayer<T>, which later handles:
    // - synchronization from local space to world space
    // - GPU upload
    // - actual rendering
}

template <typename T>
typename TriangleMeshLayer<T>::Builder
TriangleMeshLayer<T>::builder() {
    // Return a fresh builder for staged TriangleMeshLayer construction.
    return Builder {};
}

template <typename T>
void
TriangleMeshLayer<T>::build_geometry(std::vector<Vector3<T>>& pos) {
    // Rebuild the local-space triangle list from scratch.
    pos.clear();

    // No bound unit means there is no source geometry to visualize.
    if (!this->_unit) return;

    // Read the geometry operator stored in the bound unit.
    const auto& q = this->_unit->geometry_operator();

    // This layer only supports triangle-mesh geometry.
    // Any other geometry type produces no output.
    if (q.type != atlas::geometry::GeometryType::TriangleMesh) return;

    // Access the concrete triangle-mesh operator from the tagged geometry union.
    auto& m = q.triangle_mesh;

    // The mesh operator must expose valid vertex and index arrays.
    //
    // vertices:
    // - array of mesh vertex positions
    //
    // indices:
    // - triplets of integer indices, one triplet per triangle
    if (!m.vertices || !m.indices) return;

    // Expand indexed mesh triangles into a flat GL_TRIANGLES vertex stream.
    //
    // For each triangle i:
    // - read its 3 vertex indices
    // - fetch the corresponding vertex positions
    // - append them in triangle order
    for (int i = 0; i < m.triangle_count; i++) {
        // Read the three vertex indices of triangle i.
        int i0 = m.indices[3 * i];
        int i1 = m.indices[3 * i + 1];
        int i2 = m.indices[3 * i + 2];

        // Append the triangle's first vertex position.
        pos.push_back(m.vertices[i0]);

        // Append the triangle's second vertex position.
        pos.push_back(m.vertices[i1]);

        // Append the triangle's third vertex position.
        pos.push_back(m.vertices[i2]);
    }
}

template <typename T>
typename TriangleMeshLayer<T>::Builder&
TriangleMeshLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& u) {
    // Stage the unit that provides the triangle-mesh geometry.
    _unit = u;
    return *this;
}

template <typename T>
void
TriangleMeshLayer<T>::Builder::validate() const {
    // A TriangleMeshLayer requires a valid source unit.
    if (!_unit) throw std::runtime_error("TriangleMeshLayer unit null");
}

template <typename T>
TriangleMeshLayer<T>
TriangleMeshLayer<T>::Builder::build() const {
    // Validate staged builder state before constructing the final layer by value.
    validate();
    return TriangleMeshLayer(_unit);
}

template <typename T>
std::shared_ptr<TriangleMeshLayer<T>>
TriangleMeshLayer<T>::Builder::make_shared() const {
    // Validate staged builder state before constructing the final layer in shared storage.
    validate();
    return std::make_shared<TriangleMeshLayer<T>>(_unit);
}

} // namespace atlas::vizkit

#endif