#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <algorithm>
#include <cmath>

namespace atlas::vizkit {

template <typename T>
CylinderLayer<T>::CylinderLayer(const atlas::UnitHostPtr<T>& unit, int slices)
    : GeometryLayer<T>(GL_TRIANGLES, unit)
    , _slices(std::max(3, slices)) { }

template <typename T>
typename CylinderLayer<T>::Builder
CylinderLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
CylinderLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    positions.clear();
    if (!this->_unit) return;

    const auto& query = this->_unit->query_operator();
    if (query.type != atlas::geometry::GeometryType::Cylinder) return;

    const auto& cyl = query.cylinder;
    if (!cyl.center || !cyl.radius || !cyl.height) return;

    const Vector3<T>& center = *cyl.center;
    const T radius = *cyl.radius;
    const T height = *cyl.height;

    if (radius <= T(0) || height <= T(0)) return;

    const T pi = static_cast<T>(3.14159265358979323846);
    const T two_pi = static_cast<T>(2) * pi;
    const T hz = height * T(0.5);

    const Vector3<T> top_center(center.x, center.y, center.z + hz);
    const Vector3<T> bottom_center(center.x, center.y, center.z - hz);

    auto ring_point = [&](T z, int i) -> Vector3<T> {
        const T u = static_cast<T>(i) / static_cast<T>(_slices);
        const T th = u * two_pi;
        return Vector3<T>(
            center.x + radius * std::cos(th),
            center.y + radius * std::sin(th),
            z);
    };

    positions.reserve(static_cast<std::size_t>(_slices) * 12);

    for (int i = 0; i < _slices; ++i) {
        const int inext = (i + 1) % _slices;

        const Vector3<T> b0 = ring_point(bottom_center.z, i);
        const Vector3<T> b1 = ring_point(bottom_center.z, inext);
        const Vector3<T> t0 = ring_point(top_center.z, i);
        const Vector3<T> t1 = ring_point(top_center.z, inext);

        positions.push_back(b0);
        positions.push_back(t0);
        positions.push_back(t1);

        positions.push_back(b0);
        positions.push_back(t1);
        positions.push_back(b1);

        positions.push_back(top_center);
        positions.push_back(t0);
        positions.push_back(t1);

        positions.push_back(bottom_center);
        positions.push_back(b1);
        positions.push_back(b0);
    }
}

template <typename T>
typename CylinderLayer<T>::Builder&
CylinderLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename CylinderLayer<T>::Builder&
CylinderLayer<T>::Builder::with_slices(int slices) noexcept {
    _slices = slices;
    return *this;
}

template <typename T>
void
CylinderLayer<T>::Builder::validate() const {
    if (!_unit) throw std::runtime_error("CylinderLayer: unit null");

    const auto& query = _unit->query_operator();
    if (query.type != atlas::geometry::GeometryType::Cylinder) {
        throw std::runtime_error("CylinderLayer: query type must be Cylinder");
    }

    if (!query.cylinder.center || !query.cylinder.radius || !query.cylinder.height) {
        throw std::runtime_error("CylinderLayer: invalid cylinder query operator");
    }

    if (*query.cylinder.radius <= T(0) || *query.cylinder.height <= T(0)) {
        throw std::runtime_error("CylinderLayer: radius and height must be positive");
    }
}

template <typename T>
CylinderLayer<T>
CylinderLayer<T>::Builder::build() const {
    validate();
    return CylinderLayer(_unit, _slices);
}

template <typename T>
std::shared_ptr<CylinderLayer<T>>
CylinderLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<CylinderLayer<T>>(_unit, _slices);
}

} // namespace atlas::vizkit

#endif