#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <algorithm>
#include <cmath>

namespace atlas::vizkit {

template <typename T>
CylinderLayer<T>::CylinderLayer(const atlas::UnitHostPtr<T>& unit, int slices)
    : GeometryLayer<T>(GL_TRIANGLES, unit)
    , _slices(std::max(3, slices)) {
    // Construct a cylinder visualization layer.
    //
    // Rendering mode:
    // - GL_TRIANGLES
    //
    // Geometry policy:
    // - the cylinder is emitted as triangle geometry
    // - the side wall is tessellated into quads, each represented by 2 triangles
    // - the top and bottom caps are emitted as triangle fans
    //
    // Slice policy:
    // - at least 3 angular slices are required to form a closed cylinder
    // - smaller user-provided values are clamped up to 3
}

template <typename T>
typename CylinderLayer<T>::Builder
CylinderLayer<T>::builder() noexcept {
    // Return a fresh builder for staged CylinderLayer construction.
    return Builder {};
}

template <typename T>
void
CylinderLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    // Rebuild the full triangle vertex buffer from scratch.
    positions.clear();

    // No unit means there is no source geometry to visualize.
    if (!this->_unit) return;

    // Read the geometry operator stored in the bound unit.
    const auto& query = this->_unit->geometry_operator();

    // This layer only supports cylinder geometry.
    // Any other geometry type produces no output.
    if (query.type != atlas::GeometryType::Cylinder) return;

    // Access the concrete cylinder operator from the tagged geometry union.
    const auto& cyl = query.cylinder;

    // The cylinder operator must expose valid center / radius / height pointers.
    if (!cyl.center || !cyl.radius || !cyl.height) return;

    // Read cylinder parameters into local variables for clarity and reuse.
    const Vector3<T>& center = *cyl.center;
    const T radius           = *cyl.radius;
    const T height           = *cyl.height;

    // A drawable cylinder requires positive radius and positive height.
    if (radius <= T(0) || height <= T(0)) return;

    // Full angular revolution in radians.
    const T two_pi = static_cast<T>(2) * pi;

    // Half-height.
    //
    // The cylinder is centered at `center`, so the top and bottom cap planes lie at:
    // - center.z + hz
    // - center.z - hz
    const T hz = height * T(0.5);

    // Explicit cap centers.
    //
    // The current visualization assumes a cylinder aligned with the z-axis in local space.
    const Vector3<T> top_center(center.x, center.y, center.z + hz);
    const Vector3<T> bottom_center(center.x, center.y, center.z - hz);

    auto ring_point = [&](T z, int i) -> Vector3<T> {
        // Compute one point on a circular ring at height z.
        //
        // Parameterization:
        // - i / _slices gives a normalized angular coordinate in [0, 1)
        // - angle = u * 2*pi
        //
        // The ring lies in the xy-plane at the provided z coordinate.
        const T u  = static_cast<T>(i) / static_cast<T>(_slices);
        const T th = u * two_pi;

        return Vector3<T>(
            center.x + radius * std::cos(th),
            center.y + radius * std::sin(th),
            z);
    };

    // Reserve enough vertices for the full mesh.
    //
    // Per slice we emit:
    // - 2 side triangles   -> 6 vertices
    // - 1 top cap triangle -> 3 vertices
    // - 1 bottom triangle  -> 3 vertices
    //
    // Total per slice:
    // - 12 vertices
    positions.reserve(static_cast<std::size_t>(_slices) * 12);

    for (int i = 0; i < _slices; ++i) {
        // Wrap to the next slice index so the final slice closes the seam.
        const int inext = (i + 1) % _slices;

        // Bottom ring points for current and next slice.
        const Vector3<T> b0 = ring_point(bottom_center.z, i);
        const Vector3<T> b1 = ring_point(bottom_center.z, inext);

        // Top ring points for current and next slice.
        const Vector3<T> t0 = ring_point(top_center.z, i);
        const Vector3<T> t1 = ring_point(top_center.z, inext);

        // --- Side wall ---
        //
        // The rectangular side patch between slice i and slice inext is split
        // into two triangles:
        // - (b0, t0, t1)
        // - (b0, t1, b1)

        positions.push_back(b0);
        positions.push_back(t0);
        positions.push_back(t1);

        positions.push_back(b0);
        positions.push_back(t1);
        positions.push_back(b1);

        // --- Top cap ---
        //
        // Emit one triangle fan piece:
        // - (top_center, t0, t1)
        //
        // Repeating this over all slices forms the full top disk.
        positions.push_back(top_center);
        positions.push_back(t0);
        positions.push_back(t1);

        // --- Bottom cap ---
        //
        // Emit one triangle fan piece:
        // - (bottom_center, b1, b0)
        //
        // Note the winding order is reversed relative to the top cap so that
        // the cap orientation remains consistent.
        positions.push_back(bottom_center);
        positions.push_back(b1);
        positions.push_back(b0);
    }
}

template <typename T>
typename CylinderLayer<T>::Builder&
CylinderLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    // Stage the source unit that provides the cylinder geometry.
    _unit = unit;
    return *this;
}

template <typename T>
typename CylinderLayer<T>::Builder&
CylinderLayer<T>::Builder::with_slices(int slices) noexcept {
    // Stage the number of angular tessellation slices.
    //
    // Final clamping to at least 3 happens in the CylinderLayer constructor.
    _slices = slices;
    return *this;
}

template <typename T>
void
CylinderLayer<T>::Builder::validate() const {
    // A CylinderLayer must have a valid source unit.
    if (!_unit) throw std::runtime_error("CylinderLayer: unit null");

    // Read the geometry operator from the staged unit.
    const auto& query = _unit->geometry_operator();

    // The bound unit must actually describe cylinder geometry.
    if (query.type != atlas::GeometryType::Cylinder) {
        throw std::runtime_error("CylinderLayer: query type must be Cylinder");
    }

    // The cylinder operator must expose valid parameter pointers.
    if (!query.cylinder.center || !query.cylinder.radius || !query.cylinder.height) {
        throw std::runtime_error("CylinderLayer: invalid cylinder query operator");
    }

    // Cylinder dimensions must be strictly positive.
    if (*query.cylinder.radius <= T(0) || *query.cylinder.height <= T(0)) {
        throw std::runtime_error("CylinderLayer: radius and height must be positive");
    }
}

template <typename T>
CylinderLayer<T>
CylinderLayer<T>::Builder::build() const {
    // Validate staged inputs before constructing the final layer by value.
    validate();
    return CylinderLayer(_unit, _slices);
}

template <typename T>
std::shared_ptr<CylinderLayer<T>>
CylinderLayer<T>::Builder::make_shared() const {
    // Validate staged inputs before constructing the final layer in shared storage.
    validate();
    return std::make_shared<CylinderLayer<T>>(_unit, _slices);
}

} // namespace atlas::vizkit

#endif