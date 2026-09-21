#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

#include <cmath>

namespace atlas::vizkit {

template <typename T>
static Vector3<T>
circle_layer_orthogonal(const Vector3<T>& n) noexcept {
    // Choose a helper axis that is unlikely to be parallel to the input normal.
    //
    // Rationale:
    // - This helper vector is later crossed with the normalized circle normal
    //   to construct an in-plane tangent basis.
    // - If the helper axis is too aligned with n, the cross product may become
    //   numerically unstable or even degenerate.
    //
    // Strategy:
    // - Prefer the global +Z axis when n is not already close to +Z / -Z.
    // - Otherwise fall back to the global +Y axis.
    if (std::abs(n.z) < static_cast<T>(0.9)) {
        return Vector3<T>(T(0), T(0), T(1));
    }

    // Use +Y when the normal is already close to the Z axis.
    return Vector3<T>(T(0), T(1), T(0));
}

template <typename T>
static Vector3<T>
circle_layer_cross(const Vector3<T>& a, const Vector3<T>& b) noexcept {
    // Return the 3D cross product a x b.
    //
    // This helper is used to construct an orthonormal basis for the circle plane:
    // - tangent   = orthogonal(helper, normal)
    // - bitangent = orthogonal(normal, tangent)
    return Vector3<T>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <typename T>
CircleLayer<T>::CircleLayer(const atlas::UnitHostPtr<T>& unit, int segments)
    : GeometryLayer<T>(GL_LINES, unit)
    , _segments(segments) {
    // Construct a circle visualization layer.
    //
    // Rendering mode:
    // - GL_LINES
    //
    // Geometry interpretation:
    // - the circle is rendered as a polyline approximation
    // - each segment contributes one line between two consecutive sampled points
    //
    // Parameters:
    // - unit     : source geometry owner
    // - segments : number of line segments used for approximation
}

template <typename T>
typename CircleLayer<T>::Builder
CircleLayer<T>::builder() noexcept {
    // Return a fresh builder for staged CircleLayer construction.
    return Builder {};
}

template <typename T>
void
CircleLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {
    // Rebuild the output line vertex list from scratch.
    positions.clear();

    // No unit means there is no geometry source to visualize.
    if (!this->_unit) return;

    // Read the geometry operator from the bound unit.
    const auto& query = this->_unit->geometry_operator();

    // This layer only supports circle geometry.
    // Any other geometry type produces no output.
    if (query.type != atlas::geometry::GeometryType::Circle) return;

    // Access the concrete circle operator stored inside the tagged geometry union.
    const auto& circle = query.circle;

    // The circle operator must expose valid center / normal / radius pointers.
    if (!circle.center || !circle.normal || !circle.radius) return;

    // A valid drawable circle requires:
    // - strictly positive radius
    // - at least 3 line segments for a closed approximation
    if (*circle.radius <= T(0) || _segments < 3) return;

    // Copy the circle normal locally so it can be normalized.
    Vector3<T> n = *circle.normal;

    // Compute squared normal length to check degeneracy before normalization.
    const T n2 = n.length_squared();

    // Degenerate normal means the circle plane is undefined.
    if (n2 <= T(0)) return;

    // Normalize the circle normal.
    n *= T(1) / static_cast<T>(std::sqrt(n2));

    // Build an in-plane tangent vector by crossing:
    // - a helper axis chosen to avoid parallel alignment
    // - the normalized circle normal
    Vector3<T> tangent = circle_layer_cross(circle_layer_orthogonal(n), n);

    // Check that the tangent is non-degenerate.
    const T t2 = tangent.length_squared();
    if (t2 <= T(0)) return;

    // Normalize the tangent.
    tangent *= T(1) / static_cast<T>(std::sqrt(t2));

    // Build the second in-plane basis vector orthogonal to both:
    // - circle normal
    // - tangent
    Vector3<T> bitangent = circle_layer_cross(n, tangent);

    // Check that the bitangent is non-degenerate.
    const T b2 = bitangent.length_squared();
    if (b2 <= T(0)) return;

    // Normalize the bitangent.
    bitangent *= T(1) / static_cast<T>(std::sqrt(b2));

    // Use a local pi constant for angle stepping.
    const T pi = static_cast<T>(3.14159265358979323846);

    // Uniform angular step for one full revolution.
    const T step = T(2) * pi / static_cast<T>(_segments);

    // Each segment contributes two endpoints in GL_LINES mode.
    positions.reserve(static_cast<std::size_t>(_segments) * 2u);

    // Generate one line segment per angular interval.
    for (int i = 0; i < _segments; ++i) {
        // Start angle of the current segment.
        const T angle0 = static_cast<T>(i) * step;

        // End angle of the current segment.
        const T angle1 = static_cast<T>(i + 1) * step;

        // Sample the start point on the circle:
        // center + radius * (cos(theta) * tangent + sin(theta) * bitangent)
        const Vector3<T> p0 =
            *circle.center
            + *circle.radius
                * (std::cos(angle0) * tangent + std::sin(angle0) * bitangent);

        // Sample the end point on the circle using the next angular step.
        const Vector3<T> p1 =
            *circle.center
            + *circle.radius
                * (std::cos(angle1) * tangent + std::sin(angle1) * bitangent);

        // Append the current line segment endpoints.
        positions.push_back(p0);
        positions.push_back(p1);
    }
}

template <typename T>
typename CircleLayer<T>::Builder&
CircleLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    // Stage the unit that provides the circle geometry.
    _unit = unit;
    return *this;
}

template <typename T>
typename CircleLayer<T>::Builder&
CircleLayer<T>::Builder::with_segments(int segments) noexcept {
    // Stage the number of line segments used for circle approximation.
    _segments = segments;
    return *this;
}

template <typename T>
void
CircleLayer<T>::Builder::validate() const {
    // A CircleLayer requires a valid source unit.
    if (!_unit) throw std::runtime_error("CircleLayer: unit null");

    // Read the geometry operator from the staged unit.
    const auto& query = _unit->geometry_operator();

    // The bound unit must actually hold circle geometry.
    if (query.type != atlas::geometry::GeometryType::Circle) {
        throw std::runtime_error("CircleLayer: query type must be Circle");
    }

    // The circle operator must expose valid pointers to all required data.
    if (!query.circle.center || !query.circle.normal || !query.circle.radius) {
        throw std::runtime_error("CircleLayer: invalid circle query operator");
    }

    // The normal must be non-zero so the circle plane is well-defined.
    if (query.circle.normal->length_squared() <= T(0)) {
        throw std::runtime_error("CircleLayer: normal must be non-zero");
    }

    // The circle radius must be strictly positive.
    if (*query.circle.radius <= T(0)) {
        throw std::runtime_error("CircleLayer: radius must be positive");
    }

    // At least 3 segments are required to form a closed polygonal approximation.
    if (_segments < 3) {
        throw std::runtime_error("CircleLayer: segments must be >= 3");
    }
}

template <typename T>
CircleLayer<T>
CircleLayer<T>::Builder::build() const {
    // Validate staged builder inputs before constructing the final layer.
    validate();

    // Construct the layer by value.
    return CircleLayer(_unit, _segments);
}

template <typename T>
std::shared_ptr<CircleLayer<T>>
CircleLayer<T>::Builder::make_shared() const {
    // Validate staged builder inputs before constructing shared ownership.
    validate();

    // Construct the layer in shared storage.
    return std::make_shared<CircleLayer<T>>(_unit, _segments);
}

} // namespace atlas::vizkit

#endif