#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
TriangleLayer<T>::TriangleLayer(const atlas::UnitHostPtr<T>& unit)
    : GeometryLayer<T>(GL_TRIANGLES, unit) {
    // Construct a triangle visualization layer.
    //
    // Rendering mode:
    // - GL_TRIANGLES
    //
    // Geometry policy:
    // - exactly one triangle is emitted
    // - the source triangle comes from the bound unit's geometry operator
    //
    // The provided unit is forwarded to GeometryLayer<T>, which manages:
    // - unit ownership/reference
    // - world-space synchronization
    // - GPU upload and rendering
}

template <typename T>
typename TriangleLayer<T>::Builder
TriangleLayer<T>::builder() {
    // Return a fresh builder for staged TriangleLayer construction.
    return Builder {};
}

template <typename T>
void
TriangleLayer<T>::build_geometry(std::vector<Vector3<T>>& pos) {
    // Rebuild the local-space triangle vertex list from scratch.
    pos.clear();

    // No bound unit means there is no source geometry to visualize.
    if (!this->_unit) return;

    // Read the geometry operator stored in the bound unit.
    const auto& q = this->_unit->geometry_operator();

    // This layer only supports single-triangle geometry.
    // Any other geometry type produces no output.
    if (q.type != atlas::geometry::GeometryType::Triangle) return;

    // Access the concrete triangle operator from the tagged geometry union.
    auto& t = q.triangle;

    // The triangle operator must expose valid pointers to all three vertices.
    if (!t.a || !t.b || !t.c) return;

    // Emit the triangle as one GL_TRIANGLES primitive.
    //
    // Vertex order:
    // - a
    // - b
    // - c
    //
    // GeometryLayer<T> will later transform these local-space vertices into
    // world-space positions if the bound unit has a non-identity transform.
    pos.push_back(*t.a);
    pos.push_back(*t.b);
    pos.push_back(*t.c);
}

template <typename T>
typename TriangleLayer<T>::Builder&
TriangleLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& u) {
    // Stage the unit that provides the triangle geometry.
    _unit = u;
    return *this;
}

template <typename T>
void
TriangleLayer<T>::Builder::validate() const {
    // A TriangleLayer requires a valid source unit.
    if (!_unit) throw std::runtime_error("TriangleLayer unit null");
}

template <typename T>
TriangleLayer<T>
TriangleLayer<T>::Builder::build() const {
    // Validate staged builder state before constructing the final layer by value.
    validate();
    return TriangleLayer(_unit);
}

template <typename T>
std::shared_ptr<TriangleLayer<T>>
TriangleLayer<T>::Builder::make_shared() const {
    // Validate staged builder state before constructing the final layer in shared storage.
    validate();
    return std::make_shared<TriangleLayer<T>>(_unit);
}

} // namespace atlas::vizkit

#endif