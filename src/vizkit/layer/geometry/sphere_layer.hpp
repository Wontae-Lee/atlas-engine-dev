#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>

namespace atlas::vizkit {

template <typename T>
SphereLayer<T>::SphereLayer(const atlas::UnitHostPtr<T>& unit, int slices, int stacks)

    : GeometryLayer<T>(GL_TRIANGLES, unit)
    , _slices(slices)
    , _stacks(stacks) { }

template <typename T>
typename SphereLayer<T>::Builder
SphereLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
SphereLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {

    positions.clear();
    if (!this->_unit) return;

    const auto& q = this->_unit->geometry_operator();
    if (q.type != atlas::geometry::GeometryType::Sphere) return;

    const auto& s = q.sphere;
    if (!s.center || !s.radius) return;

    Vector3<T> c = *s.center;
    T r          = *s.radius;

    const T pi = 3.14159265358979323846;

    for (int i = 0; i < _stacks; i++) {
        for (int j = 0; j < _slices; j++) {
            T u = j / (T)_slices;
            T v = i / (T)_stacks;

            T th = u * 2 * pi;
            T ph = v * pi;

            positions.emplace_back(
                c.x + r * sin(ph) * cos(th),
                c.y + r * cos(ph),
                c.z + r * sin(ph) * sin(th));
        }
    }
}

template <typename T>
typename SphereLayer<T>::Builder&
SphereLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename SphereLayer<T>::Builder&
SphereLayer<T>::Builder::with_slices(int s) noexcept {
    _slices = s;
    return *this;
}

template <typename T>
typename SphereLayer<T>::Builder&
SphereLayer<T>::Builder::with_stacks(int s) noexcept {
    _stacks = s;
    return *this;
}

template <typename T>
void
SphereLayer<T>::Builder::validate() const {

    if (!_unit) throw std::runtime_error("SphereLayer: unit null");
}

template <typename T>
SphereLayer<T>
SphereLayer<T>::Builder::build() const {
    validate();
    return SphereLayer(_unit, _slices, _stacks);
}

template <typename T>
std::shared_ptr<SphereLayer<T>>
SphereLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<SphereLayer<T>>(_unit, _slices, _stacks);
}

}

#endif
