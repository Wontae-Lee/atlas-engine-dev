#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
TriangleLayer<T>::TriangleLayer(const atlas::UnitHostPtr<T>& unit)
    // GL_TRIANGLES interprets each consecutive triple of vertices as one
    // independent triangle. Because this layer renders exactly one triangle,
    // the generated vertex stream contains exactly three positions.
    : GeometryLayer<T>(GL_TRIANGLES, unit) { }

template <typename T>
typename TriangleLayer<T>::Builder
TriangleLayer<T>::builder() {
    return Builder {};
}

template <typename T>
void
TriangleLayer<T>::build_geometry(std::vector<Vector3<T>>& pos) {

    // Build the primitive in local space first. The base GeometryLayer later
    // transforms these points into world space and submits them to OpenGL.
    pos.clear();
    if (!this->_unit) return;

    auto& q = this->_unit->query_operator();

    // This layer is specialized to the triangle query variant only.
    if (q.type != atlas::geometry::GeometryType::Triangle) return;

    auto& t = q.triangle;
    if (!t.a || !t.b || !t.c) return;

    // Emit one explicit triangle:
    //   p0 = a
    //   p1 = b
    //   p2 = c
    //
    // In OpenGL triangle-list rendering, these three vertices are consumed as a
    // single rasterized primitive by glDrawArrays(GL_TRIANGLES, ...).
    //
    // The order is preserved exactly as provided by the query operator. That
    // order determines the triangle winding, which can matter for:
    // - front/back face classification
    // - back-face culling
    // - geometric normal direction if later inferred by cross products such as
    //     n = normalize((b - a) x (c - a))
    pos.push_back(*t.a);
    pos.push_back(*t.b);
    pos.push_back(*t.c);
}

template <typename T>
typename TriangleLayer<T>::Builder&
TriangleLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& u) {
    _unit = u;
    return *this;
}

template <typename T>
void
TriangleLayer<T>::Builder::validate() const {
    // Require a source unit up front so layer construction fails early instead
    // of producing an object with no geometry source.
    if (!_unit) throw std::runtime_error("TriangleLayer unit null");
}

template <typename T>
TriangleLayer<T>
TriangleLayer<T>::Builder::build() const {
    validate();
    return TriangleLayer(_unit);
}

template <typename T>
std::shared_ptr<TriangleLayer<T>>
TriangleLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<TriangleLayer<T>>(_unit);
}

}

#endif
