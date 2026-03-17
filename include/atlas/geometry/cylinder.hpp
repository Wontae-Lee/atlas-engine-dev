#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace atlas::geometry {

/* ====================================================================== */
/* Cylinder<T>                                                             */
/* ====================================================================== */

template <typename T>
Cylinder<T>::Cylinder() noexcept
    : center(T(0), T(0), T(0))
    , radius(T(1))
    , height(T(1)) {
    // Default constructor creates a canonical cylinder:
    // - center at origin
    // - radius = 1
    // - height = 1
    //
    // Coordinate convention (as used by the trace/query operators):
    // - Cylinder is axis-aligned along Z in its local space.
    // - Its caps lie at z = center.z ± height/2.
    //
    // Rationale:
    // - Provides a non-degenerate, immediately usable primitive.
}

template <typename T>
Cylinder<T>::Cylinder(const Vector3<T>& center_, T radius_, T height_) noexcept
    : center(center_)
    , radius(radius_)
    , height(height_) {
    // Construct cylinder directly from parameters.
    //
    // Caller responsibility:
    // - This constructor does NOT validate radius/height.
    // - If radius <= 0 or height <= 0, the cylinder becomes degenerate/invalid.
    //
    // If you want enforced validity, use:
    //   Cylinder<T>::builder().with_center(...).with_radius(...).with_height(...).build()
}

template <typename T>
typename Cylinder<T>::Builder
Cylinder<T>::builder() noexcept {
    // Builder entry point.
    //
    // Why a builder?
    // - Enables validation (radius/height > 0, finite values, etc.) before constructing.
    // - Keeps call sites readable when parameters are set in multiple steps.
    return Builder {};
}

template <typename T>
TraceOperator<T>
Cylinder<T>::make_trace_operator() const {
    // Construct a TraceOperator for ray-cylinder intersection tests.
    //
    // Implementation approach:
    // - Create a POD-like CylinderTraceOperator<T>.
    // - Provide it raw pointers to this cylinder's parameters.
    //
    // Why raw pointers?
    // - Operators are commonly copied into kernels; keeping them small matters.
    // - Storing pointers lets the operator read the latest parameters without
    //   duplicating state.
    //
    // Lifetime requirement:
    // - The returned operator contains pointers to members of *this*.
    // - Therefore, the Cylinder<T> object must outlive all uses of the operator.
    atlas::spatial::CylinderTraceOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return TraceOperator<T>(op);
}

template <typename T>
QueryOperator<T>
Cylinder<T>::make_query_operator() const {
    // Construct a QueryOperator for closest-point/normal/distance queries.
    //
    // Same pointer/lifetime considerations as make_trace_operator().
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return QueryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute closest point on (or inside) the cylinder to query point p.
    //
    // Delegation pattern:
    // - Create a temporary query operator wired to this cylinder's parameters.
    // - Reuse the operator's implementation to keep behavior consistent across APIs.
    //
    // Cost:
    // - The operator is tiny (just pointers), so this is cheap.
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute outward normal of the closest surface feature to p.
    //
    // Expected behavior (typical):
    // - Outside near side wall: normal points radially outward (x,y,0) normalized.
    // - Outside near cap: normal is (0,0,±1).
    // - Inside: usually normal of nearest surface (tie-breaks on edges).
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.closest_normal(p);
}

template <typename T>
T
Cylinder<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to the cylinder (SDF).
    //
    // Typical convention:
    // - negative inside
    // - zero on the surface
    // - positive outside
    //
    // Delegates to CylinderQueryOperator<T> for a single source of truth.
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.signed_distance(p);
}

template <typename T>
bool
Cylinder<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Keep finite-cylinder interior classification centralized in the query
    // operator so tolerance handling is shared across all call paths.
    return make_query_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Cylinder<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-band checks to the query operator to avoid duplicating
    // cap/side-wall boundary logic in the geometry wrapper.
    return make_query_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Cylinder<T>::centroid() const noexcept {
    // Cylinder centroid in this representation is its center.
    //
    // Delegation ensures consistent definition with query operator.
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Cylinder<T>::bound() const noexcept {
    // Axis-aligned bounding box of this cylinder.
    //
    // Since the cylinder is axis-aligned along Z:
    // - x,y extents are center ± radius
    // - z extent is center.z ± height/2
    //
    // Delegation keeps bounding behavior consistent with other primitives.
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.bound();
}

template <typename T>
bool
Cylinder<T>::is_valid() const noexcept {
    // Validate cylinder parameters.
    //
    // Typical validity rules:
    // - radius > 0
    // - height > 0
    // - values are finite
    //
    // Exact rules are defined in CylinderQueryOperator<T>::is_valid().
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&center);
    op.radius = atlas::raw_pointer_cast(&radius);
    op.height = atlas::raw_pointer_cast(&height);
    return op.is_valid();
}

template <typename T>
GeometryType
Cylinder<T>::type() const noexcept {
    // Return the geometry type tag for this class.
    return GeometryType::Cylinder;
}

/* ====================================================================== */
/* Cylinder<T>::Builder                                                    */
/* ====================================================================== */

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_center(const Vector3<T>& center_) noexcept {
    // Set center of the cylinder.
    //
    // The cylinder axis is assumed Z-aligned; center defines mid-point along Z.
    _center = center_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_radius(T radius_) noexcept {
    // Set cylinder radius in XY plane.
    //
    // Valid cylinders require radius > 0 (enforced in validate()).
    _radius = radius_;
    return *this;
}

template <typename T>
typename Cylinder<T>::Builder&
Cylinder<T>::Builder::with_height(T height_) noexcept {
    // Set cylinder height along Z.
    //
    // Valid cylinders require height > 0 (enforced in validate()).
    _height = height_;
    return *this;
}

template <typename T>
void
Cylinder<T>::Builder::validate() const {
    // Validate builder parameters prior to building Cylinder<T>.
    //
    // Rule set is delegated to CylinderQueryOperator<T>::is_valid():
    // - This keeps validity rules consistent across the codebase.
    //
    // Implementation:
    // - Create a temporary query operator pointing to builder-owned fields.
    // - Safe because validate() uses them immediately.
    atlas::geometry::CylinderQueryOperator<T> op;
    op.center = atlas::raw_pointer_cast(&_center);
    op.radius = atlas::raw_pointer_cast(&_radius);
    op.height = atlas::raw_pointer_cast(&_height);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Cylinder::Builder validation failed: radius and height must be > 0, and values must be finite.";
        throw std::runtime_error("Cylinder::Builder: invalid parameters.");
    }
}

template <typename T>
Cylinder<T>
Cylinder<T>::Builder::build() const {
    // Build a Cylinder<T> after validation.
    //
    // Strong exception guarantee:
    // - If validate() throws, no Cylinder is produced.
    validate();

    Cylinder<T> c {};

    // Copy validated parameters into the final cylinder object.
    c.center = _center;
    c.radius = _radius;
    c.height = _height;

    return c;
}

template <typename T>
atlas::host_shared_ptr<Cylinder<T>>
Cylinder<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - Build by value
    // - Move into a shared, heap-allocated cylinder object
    auto c = build();
    return atlas::make_host_shared<Cylinder<T>>(std::move(c));
}

} // namespace atlas::geometry
