#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
TriangleLayer<T>::TriangleLayer(const atlas::UnitHostPtr<T>& unit)

    : GeometryLayer<T>(GL_TRIANGLES, unit) { }

template <typename T>
typename TriangleLayer<T>::Builder
TriangleLayer<T>::builder() {
    return Builder {};
}

template <typename T>
void
TriangleLayer<T>::build_geometry(std::vector<Vector3<T>>& pos) {

    pos.clear();
    if (!this->_unit) return;

    auto& q = this->_unit->query_operator();

    if (q.type != atlas::geometry::GeometryType::Triangle) return;

    auto& t = q.triangle;
    if (!t.a || !t.b || !t.c) return;

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