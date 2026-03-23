#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>

namespace atlas::vizkit {

template <typename T>
static Vector3<T>
circle_layer_orthogonal(const Vector3<T>& n) noexcept {
    if (std::abs(n.z) < static_cast<T>(0.9)) {
        return Vector3<T>(T(0), T(0), T(1));
    }
    return Vector3<T>(T(0), T(1), T(0));
}

template <typename T>
static Vector3<T>
circle_layer_cross(const Vector3<T>& a, const Vector3<T>& b) noexcept {
    return Vector3<T>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <typename T>
CircleLayer<T>::CircleLayer(const atlas::UnitHostPtr<T>& unit, int segments)
    : GeometryLayer<T>(GL_LINES, unit)
    , _segments(segments) { }

template <typename T>
typename CircleLayer<T>::Builder
CircleLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
CircleLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    positions.clear();
    if (!this->_unit) return;

    const auto& query = this->_unit->geometry_operator();
    if (query.type != atlas::geometry::GeometryType::Circle) return;

    const auto& circle = query.circle;
    if (!circle.center || !circle.normal || !circle.radius) return;
    if (*circle.radius <= T(0) || _segments < 3) return;

    Vector3<T> n = *circle.normal;
    const T n2   = n.length_squared();
    if (n2 <= T(0)) return;
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    Vector3<T> tangent = circle_layer_cross(circle_layer_orthogonal(n), n);
    const T t2         = tangent.length_squared();
    if (t2 <= T(0)) return;
    tangent *= T(1) / static_cast<T>(std::sqrt(t2));

    Vector3<T> bitangent = circle_layer_cross(n, tangent);
    const T b2           = bitangent.length_squared();
    if (b2 <= T(0)) return;
    bitangent *= T(1) / static_cast<T>(std::sqrt(b2));

    const T pi = static_cast<T>(3.14159265358979323846);
    const T step = T(2) * pi / static_cast<T>(_segments);

    positions.reserve(static_cast<std::size_t>(_segments) * 2u);
    for (int i = 0; i < _segments; ++i) {
        const T angle0 = static_cast<T>(i) * step;
        const T angle1 = static_cast<T>(i + 1) * step;

        const Vector3<T> p0 =
            *circle.center + *circle.radius * (std::cos(angle0) * tangent + std::sin(angle0) * bitangent);
        const Vector3<T> p1 =
            *circle.center + *circle.radius * (std::cos(angle1) * tangent + std::sin(angle1) * bitangent);

        positions.push_back(p0);
        positions.push_back(p1);
    }
}

template <typename T>
typename CircleLayer<T>::Builder&
CircleLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename CircleLayer<T>::Builder&
CircleLayer<T>::Builder::with_segments(int segments) noexcept {
    _segments = segments;
    return *this;
}

template <typename T>
void
CircleLayer<T>::Builder::validate() const {
    if (!_unit) throw std::runtime_error("CircleLayer: unit null");

    const auto& query = _unit->geometry_operator();
    if (query.type != atlas::geometry::GeometryType::Circle) {
        throw std::runtime_error("CircleLayer: query type must be Circle");
    }

    if (!query.circle.center || !query.circle.normal || !query.circle.radius) {
        throw std::runtime_error("CircleLayer: invalid circle query operator");
    }

    if (query.circle.normal->length_squared() <= T(0)) {
        throw std::runtime_error("CircleLayer: normal must be non-zero");
    }

    if (*query.circle.radius <= T(0)) {
        throw std::runtime_error("CircleLayer: radius must be positive");
    }

    if (_segments < 3) {
        throw std::runtime_error("CircleLayer: segments must be >= 3");
    }
}

template <typename T>
CircleLayer<T>
CircleLayer<T>::Builder::build() const {
    validate();
    return CircleLayer(_unit, _segments);
}

template <typename T>
std::shared_ptr<CircleLayer<T>>
CircleLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<CircleLayer<T>>(_unit, _segments);
}

} // namespace atlas::vizkit

#endif
