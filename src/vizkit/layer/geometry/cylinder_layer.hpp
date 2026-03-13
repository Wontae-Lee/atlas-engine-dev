#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <algorithm>
#include <cmath>

namespace atlas::vizkit {

template <typename T>
CylinderLayer<T>::CylinderLayer(const atlas::UnitHostPtr<T>& unit, int slices)
    // GL_TRIANGLES means every 3 consecutive vertices form one triangle.
    // A tessellated cylinder is therefore emitted as a flat triangle list.
    : GeometryLayer<T>(GL_TRIANGLES, unit)
    // At least 3 slices are required to form a closed polygonal approximation
    // of the circular cross section. Fewer would be degenerate.
    , _slices(std::max(3, slices)) { }

template <typename T>
typename CylinderLayer<T>::Builder
CylinderLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
CylinderLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    // Build the cylinder in local object coordinates. The base class later
    // handles synchronization into world space and OpenGL upload.
    positions.clear();
    if (!this->_unit) return;

    const auto& query = this->_unit->query_operator();
    if (query.type != atlas::geometry::GeometryType::Cylinder) return;

    const auto& cyl = query.cylinder;
    if (!cyl.center || !cyl.radius || !cyl.height) return;

    const Vector3<T>& center = *cyl.center;
    const T radius           = *cyl.radius;
    const T height           = *cyl.height;

    // A geometric cylinder requires positive radius and positive height.
    // Zero or negative values collapse the surface or invert meaning.
    if (radius <= T(0) || height <= T(0)) return;
    const T two_pi = static_cast<T>(2) * pi;

    // hz is the half-height so the cylinder spans symmetrically around center:
    //   z_top    = center.z + height/2
    //   z_bottom = center.z - height/2
    const T hz = height * T(0.5);

    const Vector3<T> top_center(center.x, center.y, center.z + hz);
    const Vector3<T> bottom_center(center.x, center.y, center.z - hz);

    auto ring_point = [&](T z, int i) -> Vector3<T> {
        // Parameterize the circular rim by angle theta in [0, 2pi).
        //
        // For a circle of radius r centered at (cx, cy):
        //   x = cx + r cos(theta)
        //   y = cy + r sin(theta)
        //
        // Using i / slices turns the continuous circle into a regular polygonal
        // approximation with equally spaced angular samples.
        const T u  = static_cast<T>(i) / static_cast<T>(_slices);
        const T th = u * two_pi;
        return Vector3<T>(
            center.x + radius * std::cos(th),
            center.y + radius * std::sin(th),
            z);
    };

    // Each slice contributes:
    // - 2 side-wall triangles
    // - 1 top-cap triangle
    // - 1 bottom-cap triangle
    // That is 4 triangles * 3 vertices = 12 vertices per slice.
    positions.reserve(static_cast<std::size_t>(_slices) * 12);

    for (int i = 0; i < _slices; ++i) {
        // Connect sample i to the next sample, wrapping around at the seam so
        // the final slice closes the cylinder.
        const int inext = (i + 1) % _slices;

        const Vector3<T> b0 = ring_point(bottom_center.z, i);
        const Vector3<T> b1 = ring_point(bottom_center.z, inext);
        const Vector3<T> t0 = ring_point(top_center.z, i);
        const Vector3<T> t1 = ring_point(top_center.z, inext);

        // Side wall as a quad split into two triangles:
        //
        //   t0 ----- t1
        //    |     / |
        //    |   /   |
        //   b0 ----- b1
        //
        // Triangle winding is kept consistent so rasterization and face culling
        // behave predictably if enabled.
        positions.push_back(b0);
        positions.push_back(t0);
        positions.push_back(t1);

        positions.push_back(b0);
        positions.push_back(t1);
        positions.push_back(b1);

        // Top cap triangle fan piece.
        // The full disk is approximated by triangles sharing top_center.
        positions.push_back(top_center);
        positions.push_back(t0);
        positions.push_back(t1);

        // Bottom cap triangle fan piece.
        // Vertex order is reversed relative to the top so the face orientation
        // remains outward-facing with a right-handed convention.
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
    // Builder validation fails early so the layer construction path does not
    // silently create an object that can never render meaningful geometry.
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
