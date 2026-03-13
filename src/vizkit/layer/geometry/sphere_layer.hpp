#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>

namespace atlas::vizkit {

template <typename T>
SphereLayer<T>::SphereLayer(const atlas::UnitHostPtr<T>& unit, int slices, int stacks)
    // The base renderer will interpret the emitted vertex stream as triangles.
    // This implementation currently samples the sphere surface parametrically,
    // so the exact visual result depends on how many vertices the shader path
    // consumes and whether higher-level code assumes a triangle-style stream.
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

    // Sample the sphere in local coordinates via angular parameters.
    positions.clear();
    if (!this->_unit) return;

    const auto& q = this->_unit->query_operator();
    if (q.type != atlas::geometry::GeometryType::Sphere) return;

    const auto& s = q.sphere;
    if (!s.center || !s.radius) return;

    Vector3<T> c = *s.center;
    T r          = *s.radius;

    // pi controls the latitude/longitude parameterization.
    const T pi = 3.14159265358979323846;

    // The nested loops sweep a regular grid over spherical coordinates:
    // - i / stacks  -> polar parameter v in [0, 1)
    // - j / slices  -> azimuth parameter u in [0, 1)
    //
    // Angles:
    //   theta = 2*pi*u   : longitude around the vertical axis
    //   phi   = pi*v     : polar angle from the +Y pole to the -Y pole
    //
    // The Cartesian conversion used here is:
    //   x = cx + r sin(phi) cos(theta)
    //   y = cy + r cos(phi)
    //   z = cz + r sin(phi) sin(theta)
    //
    // This is the standard spherical-coordinate map with Y treated as the pole
    // axis instead of Z.
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
    // The current builder enforces only that a unit exists. More detailed
    // geometric validation remains inside the runtime generation path.
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
