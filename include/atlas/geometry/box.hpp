#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace atlas::geometry {

/* ====================================================================== */
/* Box<T>                                                                  */
/* ====================================================================== */

template <typename T>
Box<T>::Box() noexcept
    : lower_corner(T(-1), T(-1), T(-1))
    , upper_corner(T(+1), T(+1), T(+1)) {
    // Default constructor creates a canonical axis-aligned box.
    //
    // Current policy:
    // - lower = (-1, -1, -1)
    // - upper = (+1, +1, +1)
    //
    // Rationale:
    // - Provides a sensible non-degenerate default volume.
    // - Keeps Box<T> immediately usable without additional setup.
    //
    // Notes:
    // - Being axis-aligned, "lower" must be component-wise <= "upper" to be valid.
}

template <typename T>
Box<T>::Box(const Vector3<T>& lower_corner_,
            const Vector3<T>& upper_corner_) noexcept
    : lower_corner(lower_corner_)
    , upper_corner(upper_corner_) {
    // Construct a box directly from bounds.
    //
    // Caller responsibility:
    // - This constructor does NOT validate bounds.
    // - If lower_corner_ has components greater than upper_corner_,
    //   the box becomes invalid for most geometric queries.
    //
    // If you want enforced validity:
    // - Use Box<T>::builder().with_params(...).build()
    //   which runs validate().
}

template <typename T>
typename Box<T>::Builder
Box<T>::builder() noexcept {
    // Entry point for fluent Builder construction.
    //
    // Builder is useful because:
    // - It can validate bounds and enforce invariants before producing Box<T>.
    // - It keeps construction readable in call sites with multiple parameters.
    return Builder {};
}

template <typename T>
TraceOperator<T>
Box<T>::make_trace_operator() const {
    // Create a TraceOperator suitable for ray/geometry intersection tests.
    //
    // Implementation strategy:
    // - Instantiate a concrete device-friendly operator (BoxTraceOperator).
    // - Provide it raw pointers to the box bounds stored in this object.
    //
    // Why raw pointers?
    // - Operator objects are typically POD-like and copied into device kernels.
    // - A raw pointer to the bounds allows the operator to read the box data
    //   without embedding large state.
    //
    // IMPORTANT lifetime note:
    // - The returned operator stores pointers to *this->lower_corner and upper_corner.
    // - Therefore, the Box<T> instance must outlive any use of the operator.
    // - This is usually safe if the geometry object is kept alive for the duration
    //   of queries/traces.
    atlas::spatial::BoxTraceOperator<T> op;

    // raw_pointer_cast:
    // - Converts address-of a host object into the pointer type expected by the operator.
    // - In some implementations, this can also handle host/device pointer wrapping.
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);

    // Wrap into the polymorphic/variant TraceOperator<T>.
    return TraceOperator<T>(op);
}

template <typename T>
QueryOperator<T>
Box<T>::make_query_operator() const {
    // Create a QueryOperator suitable for closest-point/normal/distance tests.
    //
    // Same lifetime and pointer considerations as make_trace_operator().
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return QueryOperator<T>(op);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute the closest point on (or in) the box to a query point p.
    //
    // Delegation approach:
    // - Build a temporary BoxQueryOperator wired to this box's bounds.
    // - Reuse the operator's tested implementation.
    //
    // Performance note:
    // - Creating the operator is very cheap (two pointers).
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Compute the outward normal of the closest surface feature to p.
    //
    // Behavior depends on operator definition:
    // - If p is outside: normal points outward from the nearest face/edge/corner.
    // - If p is inside : normal is typically the normal of the nearest face (tie-breaking
    //   rules may apply on edges/corners).
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.closest_normal(p);
}

template <typename T>
T
Box<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Signed distance to the box.
    //
    // Common convention (typical SDF for an AABB):
    // - Negative inside, zero on surface, positive outside.
    //
    // Delegates to the query operator for consistent behavior across the codebase.
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.signed_distance(p);
}

template <typename T>
bool
Box<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward host-side inside classification through the query operator so
    // Box<T> and BoxQueryOperator<T> keep identical tolerance semantics.
    return make_query_operator().is_inside(p, tolerance);
}

template <typename T>
bool
Box<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Surface-band classification is delegated to the query operator for
    // consistency with all other box query entry points.
    return make_query_operator().is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Box<T>::centroid() const noexcept {
    // Return the box centroid:
    //   c = (lower + upper) / 2
    //
    // Delegation ensures identical behavior between Box<T> and BoxQueryOperator<T>.
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Box<T>::bound() const noexcept {
    // Return the axis-aligned bounding box of this box.
    //
    // For an axis-aligned box geometry, the bound is the box itself.
    // Still delegated for API uniformity with other geometry types.
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.bound();
}

template <typename T>
bool
Box<T>::is_valid() const noexcept {
    // Validate bounds:
    // - A box is valid if lower_corner <= upper_corner component-wise.
    //
    // Delegation keeps the definition of validity centralized in BoxQueryOperator<T>.
    atlas::geometry::BoxQueryOperator<T> op;
    op.lower_corner = atlas::raw_pointer_cast(&lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&upper_corner);
    return op.is_valid();
}

template <typename T>
GeometryType
Box<T>::type() const noexcept {
    // Return the geometry type tag for this box.
    return GeometryType::Box;
}

/* ====================================================================== */
/* Box<T>::Builder                                                         */
/* ====================================================================== */

template <typename T>
Box<T>
Box<T>::Builder::build() const {
    // Build a Box<T> after validation.
    //
    // Strong exception guarantee:
    // - If validate() throws, no Box is returned.
    validate();

    Box<T> b {};

    // Copy builder state into the final object.
    b.lower_corner = _lower_corner;
    b.upper_corner = _upper_corner;

    return b;
}

template <typename T>
atlas::host_shared_ptr<Box<T>>
Box<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - Build the Box<T> by value
    // - Move it into a shared, heap-allocated object
    auto b = build();
    return atlas::make_host_shared<Box<T>>(std::move(b));
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_lower_corner(const Vector3<T>& lower_corner_) noexcept {
    // Set lower corner (min corner) of the box.
    //
    // No validation here:
    // - Builder collects inputs; validation happens in validate()/build().
    _lower_corner = lower_corner_;
    return *this;
}

template <typename T>
typename Box<T>::Builder&
Box<T>::Builder::with_upper_corner(const Vector3<T>& upper_corner_) noexcept {
    // Set upper corner (max corner) of the box.
    _upper_corner = upper_corner_;
    return *this;
}

template <typename T>
void
Box<T>::Builder::validate() const {
    // Validate builder state before constructing Box<T>.
    //
    // Rule:
    // - lower_corner must be component-wise <= upper_corner.
    //
    // We reuse BoxQueryOperator<T>::is_valid() so "validity" is defined once.
    atlas::geometry::BoxQueryOperator<T> op;

    // Here we point at builder-owned temporary storage (_lower_corner/_upper_corner).
    // That is safe because validate() uses them immediately.
    op.lower_corner = atlas::raw_pointer_cast(&_lower_corner);
    op.upper_corner = atlas::raw_pointer_cast(&_upper_corner);

    if (!op.is_valid()) {
        atlas::logger::error()
            << "Box::Builder validation failed: lower_corner must be <= upper_corner.";
        throw std::runtime_error("Box::Builder: invalid parameters.");
    }
}

} // namespace atlas::geometry
