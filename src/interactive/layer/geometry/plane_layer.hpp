#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>

namespace atlas::vizkit {

template <typename T>
static Vector3<T>
plane_layer_cross(const Vector3<T>& a, const Vector3<T>& b) noexcept {
    // Return the 3D cross product a x b.
    //
    // This helper is used to construct an orthonormal basis on the plane:
    // - one tangent axis lying on the plane
    // - one bitangent axis also lying on the plane
    return Vector3<T>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <typename T>
PlaneLayer<T>::PlaneLayer(const atlas::UnitHostPtr<T>& unit, T extent)
    : GeometryLayer<T>(GL_TRIANGLES, unit)
    , _extent(extent) {
    // Construct a plane visualization layer.
    //
    // Rendering mode:
    // - GL_TRIANGLES
    //
    // Geometry policy:
    // - the infinite analytic plane is visualized as a finite quad
    // - that quad is centered on one point lying on the plane
    // - the quad size is controlled by _extent
    //
    // The quad is emitted as:
    // - 2 triangles
    // - 6 vertices total
}

template <typename T>
typename PlaneLayer<T>::Builder
PlaneLayer<T>::builder() noexcept {
    // Return a fresh builder for staged PlaneLayer construction.
    return Builder {};
}

template <typename T>
void
PlaneLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    // Rebuild the full local-space triangle geometry from scratch.
    positions.clear();

    // No unit means there is no source geometry to visualize.
    if (!this->_unit) return;

    // Read the geometry operator from the bound unit.
    const auto& query = this->_unit->geometry_operator();

    // This layer only supports plane geometry.
    // Any other geometry type produces no output.
    if (query.type != atlas::geometry::GeometryType::Plane) return;

    // Access the concrete plane operator from the tagged geometry union.
    const auto& plane = query.plane;

    // The plane operator must expose valid normal and offset pointers.
    if (!plane.normal || !plane.offset) return;

    // Copy the plane normal locally so it can be normalized.
    Vector3<T> n = *plane.normal;

    // Check whether the plane normal is degenerate before normalization.
    const T n2 = n.length_squared();
    if (n2 <= T(0)) return;

    // Normalize the plane normal.
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    // Read the plane offset from the plane equation:
    //   dot(n, x) + d = 0
    const T d = *plane.offset;

    // Compute one point lying on the plane.
    //
    // For normalized n, the point:
    //   p0 = -d * n
    // satisfies:
    //   dot(n, p0) + d = 0
    const Vector3<T> p0 = -d * n;

    // Choose a reference axis that is unlikely to be parallel to the plane normal.
    //
    // Why:
    // - we need a stable vector to cross with n in order to build a tangent axis
    // - if the reference vector is too aligned with n, the cross product becomes
    //   degenerate or numerically unstable
    const Vector3<T> ref = (std::abs(n.z) < static_cast<T>(0.9))
        ? Vector3<T>(T(0), T(0), T(1))
        : Vector3<T>(T(0), T(1), T(0));

    // Build the first in-plane tangent axis.
    //
    // Since u = ref x n, u is perpendicular to n and therefore lies on the plane.
    Vector3<T> u = plane_layer_cross(ref, n);

    // Validate the tangent axis before normalization.
    const T u2 = u.length_squared();
    if (u2 <= T(0)) return;

    // Normalize the tangent axis.
    u *= T(1) / static_cast<T>(std::sqrt(u2));

    // Build the second in-plane axis orthogonal to both n and u.
    //
    // Since v = n x u, v is also tangent to the plane.
    Vector3<T> v = plane_layer_cross(n, u);

    // Normalize the second tangent axis when valid.
    const T v2 = v.length_squared();
    if (v2 > T(0)) {
        v *= T(1) / static_cast<T>(std::sqrt(v2));
    }

    // Read the half-extent of the finite quad used to visualize the plane.
    //
    // The final quad corners are constructed as:
    //   p0 ± e*u ± e*v
    const T e = _extent;

    // Construct the four quad corners around the anchor point p0.
    //
    // Corner naming:
    // - p00 : (-u, -v)
    // - p10 : (+u, -v)
    // - p11 : (+u, +v)
    // - p01 : (-u, +v)
    const Vector3<T> p00 = p0 - e * u - e * v;
    const Vector3<T> p10 = p0 + e * u - e * v;
    const Vector3<T> p11 = p0 + e * u + e * v;
    const Vector3<T> p01 = p0 - e * u + e * v;

    // A quad rendered as triangles requires exactly 6 vertices.
    positions.reserve(6);

    // Emit the first triangle of the quad.
    //
    // Triangle:
    //   p00 -> p10 -> p11
    positions.push_back(p00);
    positions.push_back(p10);
    positions.push_back(p11);

    // Emit the second triangle of the quad.
    //
    // Triangle:
    //   p00 -> p11 -> p01
    positions.push_back(p00);
    positions.push_back(p11);
    positions.push_back(p01);
}

template <typename T>
typename PlaneLayer<T>::Builder&
PlaneLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    // Stage the source unit that provides the plane geometry.
    _unit = unit;
    return *this;
}

template <typename T>
typename PlaneLayer<T>::Builder&
PlaneLayer<T>::Builder::with_extent(T extent) noexcept {
    // Stage the finite visualization extent of the plane quad.
    _extent = extent;
    return *this;
}

template <typename T>
void
PlaneLayer<T>::Builder::validate() const {
    // A PlaneLayer must have a valid source unit.
    if (!_unit) throw std::runtime_error("PlaneLayer: unit null");

    // Read the geometry operator from the staged unit.
    const auto& query = _unit->geometry_operator();

    // The bound unit must actually contain plane geometry.
    if (query.type != atlas::geometry::GeometryType::Plane) {
        throw std::runtime_error("PlaneLayer: query type must be Plane");
    }

    // The plane operator must expose valid normal and offset pointers.
    if (!query.plane.normal || !query.plane.offset) {
        throw std::runtime_error("PlaneLayer: invalid plane query operator");
    }

    // The plane normal must be non-zero.
    if (query.plane.normal->length_squared() <= T(0)) {
        throw std::runtime_error("PlaneLayer: normal must be non-zero");
    }

    // The finite plane extent must be strictly positive.
    if (_extent <= T(0)) {
        throw std::runtime_error("PlaneLayer: extent must be positive");
    }
}

template <typename T>
PlaneLayer<T>
PlaneLayer<T>::Builder::build() const {
    // Validate staged inputs before constructing the final layer by value.
    validate();
    return PlaneLayer(_unit, _extent);
}

template <typename T>
std::shared_ptr<PlaneLayer<T>>
PlaneLayer<T>::Builder::make_shared() const {
    // Validate staged inputs before constructing the final layer in shared storage.
    validate();
    return std::make_shared<PlaneLayer<T>>(_unit, _extent);
}

} // namespace atlas::vizkit

#endif