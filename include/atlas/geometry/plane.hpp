#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Plane<T>::Plane() noexcept
    : normal(T(0), T(0), T(1))
    , offset(T(0)) {
    // Construct a default plane whose normal points along +z and whose offset is zero.
    //
    // This corresponds to the plane:
    //     z = 0
    //
    // After initializing the geometric parameters, bind the cached operator so all
    // delegated geometry queries reference this object's member storage.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& normal_, T offset_) noexcept
    : normal(normal_)
    , offset(offset_) {
    // Construct a plane directly from the implicit plane equation parameters.
    //
    // The stored representation follows:
    //     n . x + d = 0
    //
    // where:
    // - `normal_` is `n`
    // - `offset_` is `d`
    //
    // Bind the cached operator so later queries see this plane's data.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept
    : normal(normal_)
    , offset(-(normal_.dot(point))) {
    // Construct a plane from one point on the plane and a normal direction.
    //
    // Starting from:
    //     n . x + d = 0
    //
    // and requiring that `point` lies on the plane gives:
    //     d = -n . point
    //
    // Bind the cached operator so delegated queries reference this object's storage.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Plane& other) noexcept
    : normal(other.normal)
    , offset(other.offset) {
    // Copy the plane parameters from another plane instance.
    //
    // Rebind the cached operator because it must point to this object's members,
    // not the source object's members.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(Plane&& other) noexcept
    : normal(std::move(other.normal))
    , offset(other.offset) {
    // Move or copy the plane parameters into this object.
    //
    // Rebind this object's cached operator to its own member storage.
    bind_operator();

    // Rebind the moved-from object's cached operator as well so its internal
    // pointers remain consistent after the move.
    other.bind_operator();
}

template <typename T>
Plane<T>&
Plane<T>::operator=(const Plane& other) noexcept {
    // Guard against self-assignment.
    if (this == &other) return *this;

    // Copy all plane parameters from the source object.
    normal = other.normal;
    offset = other.offset;

    // Rebind the cached operator so it continues to reference this object's members.
    bind_operator();
    return *this;
}

template <typename T>
Plane<T>&
Plane<T>::operator=(Plane&& other) noexcept {
    // Guard against self-move-assignment.
    if (this == &other) return *this;

    // Move or copy the plane parameters from the source object.
    normal = std::move(other.normal);
    offset = other.offset;

    // Rebind this object's cached operator to its current members.
    bind_operator();

    // Rebind the moved-from object's cached operator so its internal pointers
    // remain aligned with its own member storage.
    other.bind_operator();
    return *this;
}

template <typename T>
void
Plane<T>::bind_operator() noexcept {
    // Bind the cached geometry operator to this plane's actual member storage.
    //
    // This must be refreshed after construction, copy, or move so all delegated
    // queries operate on the correct `normal` and `offset`.
    _operator.normal = atlas::raw_pointer_cast(&normal);
    _operator.offset = atlas::raw_pointer_cast(&offset);
}

template <typename T>
typename Plane<T>::Builder
Plane<T>::builder() noexcept {
    // Return a fresh builder object for staged plane construction.
    //
    // The builder path is useful when the normal and offset, or point and normal,
    // are supplied incrementally before final validation and construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Plane<T>::make_geometry_operator() const {
    // Wrap the cached plane-specific operator in the generic geometry-operator
    // interface used elsewhere in the geometry system.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-point query to the cached bound operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the closest-normal query to the cached bound operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Plane<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Forward the signed-distance query to the cached bound operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Plane<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward inside classification to the cached bound operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Plane<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Forward surface classification to the cached bound operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::centroid() const noexcept {
    // Forward centroid computation to the cached bound operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Plane<T>::bound() const noexcept {
    // Forward bounding-box computation to the cached bound operator.
    return _operator.bound();
}

template <typename T>
bool
Plane<T>::is_valid() const noexcept {
    // Forward validity testing to the cached bound operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Plane<T>::type() const noexcept {
    // Return the runtime geometry type tag for this concrete geometry.
    return GeometryType::Plane;
}

template <typename T>
Plane<T>
Plane<T>::Builder::build() const {
    // Validate all staged builder parameters before constructing the plane.
    validate();

    // Start from a default-constructed plane so its cached operator is already
    // bound to its own member storage.
    Plane<T> p {};

    // Overwrite the default plane parameters with the validated builder state.
    p.normal = _normal;
    p.offset = _offset;

    // No explicit rebind is required because the cached operator already points
    // to `p`'s own member storage, and only the stored values changed.
    return p;
}

template <typename T>
atlas::host_shared_ptr<Plane<T>>
Plane<T>::Builder::make_host_shared() const {
    // Build the plane by value first.
    auto p = build();

    // Move the built plane into host-shared managed storage.
    return atlas::make_host_shared<Plane<T>>(std::move(p));
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store the plane normal in the builder's staged state.
    _normal = normal_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_offset(T offset_) noexcept {
    // Store the plane offset term in the builder's staged state.
    //
    // The implicit equation remains:
    //     n . x + d = 0
    //
    // where `_offset` is `d`.
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept {
    // Store both implicit plane-equation parameters at once.
    _normal = normal_;
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_point_normal(const Vector3<T>& point,
                                     const Vector3<T>& normal_) noexcept {
    // Build the implicit plane representation from a point and a normal.
    //
    // Using:
    //     n . x + d = 0
    // and requiring `point` to lie on the plane gives:
    //     d = -n . point
    _normal = normal_;
    _offset = -(normal_.dot(point));
    return *this;
}

template <typename T>
void
Plane<T>::Builder::validate() const {
    // Reuse the runtime geometry-operator validity logic so the definition
    // of a valid plane remains centralized in one place.
    atlas::geometry::PlaneGeometryOperator<T> op;
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.offset = atlas::raw_pointer_cast(&_offset);

    // Reject invalid staged parameters with both a log message and an exception.
    if (!op.is_valid()) {
        atlas::logger::error()
            << "Plane::Builder validation failed: normal must be finite and non-zero; offset must be finite.";
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, return the query point unchanged.
    if (!normal || !offset) return p;

    // Compute the signed deviation from the plane:
    //     sdev = n . p + d
    //
    // This assumes `normal` is already stored in the intended scale for
    // projection and signed-distance interpretation.
    const T sdev = ((*normal).dot(p) + (*offset));

    // Orthogonally project `p` onto the plane by subtracting the signed
    // deviation along the plane normal.
    return p - sdev * (*normal);
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // If the normal pointer is missing, return the zero vector as a safe fallback.
    if (!normal) return atlas::math::Vector<T, 3>(T(0), T(0), T(0));

    // For a plane, the surface normal is constant everywhere.
    return *normal;
}

template <typename T>
T
PlaneGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // If the operator is not fully bound, report an infinite distance so the
    // query can be treated as invalid by the caller.
    if (!normal || !offset) return std::numeric_limits<T>::infinity();

    // Evaluate the implicit plane equation:
    //     n . p + d
    //
    // When `normal` is unit length, this is the usual signed distance.
    // Otherwise it is a scaled signed distance consistent with the stored normal.
    return (*normal).dot(p) + (*offset);
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // If the operator is not fully bound, containment cannot be established.
    if (!normal || !offset) return false;

    // Interpret the "inside" region as the half-space:
    //     n . p + d <= tolerance
    //
    // With zero tolerance, this is the negative side of the plane together
    // with the plane itself.
    return (*normal).dot(p) + (*offset) <= tolerance;
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // A point is considered on the plane when its absolute signed distance
    // lies within the specified tolerance band around zero.
    return std::abs(signed_distance(p)) <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::centroid() const noexcept {
    // A plane is unbounded and therefore has no true finite centroid.
    //
    // Return the origin as a conventional neutral placeholder.
    return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
PlaneGeometryOperator<T>::bound() const noexcept {
    // A plane is unbounded, so represent its axis-aligned bound as the widest
    // finite box expressible by the scalar type.
    //
    // This is a practical placeholder rather than a mathematically exact finite bound.
    const T lo = std::numeric_limits<T>::lowest();
    const T hi = std::numeric_limits<T>::max();

    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>(lo, lo, lo),
        atlas::math::Vector<T, 3>(hi, hi, hi));
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_valid() const noexcept {
    // Both pointers must be present.
    if (!normal || !offset) return false;

    // A valid plane requires:
    // - a non-zero normal vector
    // - a finite offset value
    //
    // Note:
    // - this implementation checks offset finiteness explicitly
    // - it only checks that the normal has non-zero length, not that each
    //   component is finite individually
    const T n2 = (*normal).length_squared();
    return (n2 > T(0)) && std::isfinite(static_cast<double>(*offset));
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Initialize the result in the default "no hit" state.
    HitSurface<T> result {};
    if (!normal || !offset) return result;

    // Alias the plane equation parameters:
    //     n . x + d = 0
    const atlas::math::Vector<T, 3>& n = *normal;
    const T d                          = *offset;

    // Compute:
    // - `denom` : how strongly the ray direction moves along the plane normal
    // - `numer` : signed plane residual of the ray origin, negated for solving t
    const T denom = n.dot(ray.direction);
    const T numer = -(n.dot(ray.origin) + d);

    if (denom == T(0)) {
        // The ray is parallel to the plane.
        if (numer != T(0)) return result;

        // The ray origin already lies on the plane, so report an immediate hit.
        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = n;
        return result;
    }

    // Solve the ray-plane intersection parameter:
    //     t = -(n . origin + d) / (n . direction)
    const T t = numer / denom;

    // Reject intersections that lie behind the ray origin.
    if (t < T(0)) return result;

    // Populate the successful hit record.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    result.normal          = n;
    return result;
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Convenience call-operator wrapper around the explicit ray-trace routine.
    return trace(ray);
}

} // namespace atlas::geometry