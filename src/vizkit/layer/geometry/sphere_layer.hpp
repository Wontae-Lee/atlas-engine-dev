#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <algorithm>
#include <cmath>

namespace atlas::vizkit {

template <typename T>
SphereLayer<T>::SphereLayer(const atlas::UnitHostPtr<T>& unit, int slices, int stacks)

    : GeometryLayer<T>(GL_LINES, unit)
    , _slices(std::max(3, slices))
    , _stacks(std::max(2, stacks)) { }

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
    if (r <= T(0)) return;

    const T two_pi = static_cast<T>(2) * pi;

    auto sphere_point = [&](int stack, int slice) -> Vector3<T> {
        const T v  = static_cast<T>(stack) / static_cast<T>(_stacks);
        const T u  = static_cast<T>(slice) / static_cast<T>(_slices);
        const T ph = v * pi;
        const T th = u * two_pi;

        return Vector3<T>(
            c.x + r * std::sin(ph) * std::cos(th),
            c.y + r * std::cos(ph),
            c.z + r * std::sin(ph) * std::sin(th));
    };

    positions.reserve(static_cast<std::size_t>(_stacks + 1) * static_cast<std::size_t>(_slices) * 4);

    for (int i = 0; i <= _stacks; ++i) {
        for (int j = 0; j < _slices; ++j) {
            const int j_next = (j + 1) % _slices;
            positions.push_back(sphere_point(i, j));
            positions.push_back(sphere_point(i, j_next));
        }
    }

    for (int i = 0; i < _stacks; ++i) {
        for (int j = 0; j < _slices; ++j) {
            positions.push_back(sphere_point(i, j));
            positions.push_back(sphere_point(i + 1, j));
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

    const auto& query = _unit->geometry_operator();
    if (query.type != atlas::geometry::GeometryType::Sphere) {
        throw std::runtime_error("SphereLayer: query type must be Sphere");
    }

    if (!query.sphere.center || !query.sphere.radius) {
        throw std::runtime_error("SphereLayer: invalid sphere query operator");
    }

    if (*query.sphere.radius <= T(0)) {
        throw std::runtime_error("SphereLayer: radius must be positive");
    }

    if (_slices < 3) {
        throw std::runtime_error("SphereLayer: slices must be >= 3");
    }

    if (_stacks < 2) {
        throw std::runtime_error("SphereLayer: stacks must be >= 2");
    }
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
