#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>

namespace atlas::vizkit {

template <typename T>
static Vector3<T>
plane_layer_cross(const Vector3<T>& a, const Vector3<T>& b) noexcept {
    // Explicit cross product implementation:
    //   a x b = (ay bz - az by, az bx - ax bz, ax by - ay bx)
    //
    // The result is perpendicular to both a and b. In this file it is used to
    // construct an orthonormal basis lying on the queried plane.
    return Vector3<T>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <typename T>
PlaneLayer<T>::PlaneLayer(const atlas::UnitHostPtr<T>& unit, T extent)
    // The plane patch is rendered as two triangles, so GL_TRIANGLES is the
    // correct OpenGL primitive mode for the generated vertex list.
    : GeometryLayer<T>(GL_TRIANGLES, unit)
    , _extent(extent) { }

template <typename T>
typename PlaneLayer<T>::Builder
PlaneLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
PlaneLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    // This function converts an infinite analytic plane into a finite square
    // patch suitable for rasterization.
    positions.clear();
    if (!this->_unit) return;

    const auto& query = this->_unit->query_operator();
    if (query.type != atlas::geometry::GeometryType::Plane) return;

    const auto& plane = query.plane;
    if (!plane.normal || !plane.offset) return;

    Vector3<T> n = *plane.normal;
    const T n2   = n.length_squared();
    if (n2 <= T(0)) return;

    // Normalize the plane normal so it has unit length.
    //
    // This matters because later formulas assume n is unit-length:
    // - p0 = -d * n becomes the closest point on the plane to the origin
    // - basis construction becomes scale-independent
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    const T d           = *plane.offset;

    // If the plane equation is interpreted as:
    //   dot(n, x) + d = 0
    // and n is unit-length, then one point on the plane is:
    //   p0 = -d * n
    //
    // Verification:
    //   dot(n, -d n) + d = -d dot(n, n) + d = -d + d = 0
    const Vector3<T> p0 = -d * n;

    // Choose a reference axis that is not almost parallel to n.
    //
    // If n were nearly parallel to the reference vector, the cross product
    // would have near-zero length and produce an unstable tangent direction.
    const Vector3<T> ref = (std::abs(n.z) < static_cast<T>(0.9))
        ? Vector3<T>(T(0), T(0), T(1))
        : Vector3<T>(T(0), T(1), T(0));

    // First tangent direction lying in the plane:
    //   u = normalize(ref x n)
    //
    // Because a cross product is perpendicular to both inputs, u is
    // perpendicular to n and therefore lies in the plane.
    Vector3<T> u = plane_layer_cross(ref, n);
    const T u2   = u.length_squared();
    if (u2 <= T(0)) return;
    u *= T(1) / static_cast<T>(std::sqrt(u2));

    // Second tangent direction:
    //   v = normalize(n x u)
    //
    // This guarantees:
    // - v is perpendicular to n
    // - v is perpendicular to u
    // so {u, v, n} becomes an orthonormal frame.
    Vector3<T> v = plane_layer_cross(n, u);
    const T v2   = v.length_squared();
    if (v2 > T(0)) {
        v *= T(1) / static_cast<T>(std::sqrt(v2));
    }

    const T e = _extent;

    // Build the four corners of a square patch centered at p0:
    //   p0 +/- e*u +/- e*v
    //
    // Geometrically, u and v span the plane, and extent scales the patch half-
    // width along each tangent axis.
    const Vector3<T> p00 = p0 - e * u - e * v;
    const Vector3<T> p10 = p0 + e * u - e * v;
    const Vector3<T> p11 = p0 + e * u + e * v;
    const Vector3<T> p01 = p0 - e * u + e * v;

    // Two triangles are sufficient to rasterize one quad in OpenGL's triangle
    // pipeline. The GPU sees only triangles, not a higher-level "plane patch".
    positions.reserve(6);

    positions.push_back(p00);
    positions.push_back(p10);
    positions.push_back(p11);

    positions.push_back(p00);
    positions.push_back(p11);
    positions.push_back(p01);
}

template <typename T>
typename PlaneLayer<T>::Builder&
PlaneLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
typename PlaneLayer<T>::Builder&
PlaneLayer<T>::Builder::with_extent(T extent) noexcept {
    _extent = extent;
    return *this;
}

template <typename T>
void
PlaneLayer<T>::Builder::validate() const {
    // Fail fast on invalid geometric inputs so the visualization layer's
    // contract is explicit instead of degrading silently at draw time.
    if (!_unit) throw std::runtime_error("PlaneLayer: unit null");

    const auto& query = _unit->query_operator();
    if (query.type != atlas::geometry::GeometryType::Plane) {
        throw std::runtime_error("PlaneLayer: query type must be Plane");
    }

    if (!query.plane.normal || !query.plane.offset) {
        throw std::runtime_error("PlaneLayer: invalid plane query operator");
    }

    if (query.plane.normal->length_squared() <= T(0)) {
        throw std::runtime_error("PlaneLayer: normal must be non-zero");
    }

    if (_extent <= T(0)) {
        throw std::runtime_error("PlaneLayer: extent must be positive");
    }
}

template <typename T>
PlaneLayer<T>
PlaneLayer<T>::Builder::build() const {
    validate();
    return PlaneLayer(_unit, _extent);
}

template <typename T>
std::shared_ptr<PlaneLayer<T>>
PlaneLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<PlaneLayer<T>>(_unit, _extent);
}

} // namespace atlas::vizkit

#endif
