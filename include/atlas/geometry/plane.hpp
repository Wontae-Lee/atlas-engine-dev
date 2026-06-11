#pragma once

#include <atlas/memory/raw_pointer_cast.h>

#include <stdexcept>
#include <utility>

namespace atlas::geometry {

template <typename T>
Plane<T>::Plane() noexcept
    : normal(T(0), T(0), T(1))
    , offset(T(0)) {
    // Bind the operator to this plane's default parameter storage.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& normal_, T offset_) noexcept
    : normal(normal_)
    , offset(offset_) {
    // Bind the operator after storing the user-provided normal and offset.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Vector3<T>& point, const Vector3<T>& normal_) noexcept
    : normal(normal_)
    , offset(-(normal_.dot(point))) {
    // Convert the point-normal representation into the implicit plane form.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(const Plane& other) noexcept
    : normal(other.normal)
    , offset(other.offset) {
    // Rebind the operator because copied raw pointers must refer to this object.
    bind_operator();
}

template <typename T>
Plane<T>::Plane(Plane&& other) noexcept
    : normal(std::move(other.normal))
    , offset(other.offset) {
    // Rebind this object after moving member storage.
    bind_operator();

    // Keep the moved-from object internally consistent.
    other.bind_operator();
}

template <typename T>
Plane<T>&
Plane<T>::operator=(const Plane& other) noexcept {
    // Avoid unnecessary rebinding on self-assignment.
    if (this == &other) {
        return *this;
    }

    normal = other.normal;
    offset = other.offset;

    // Rebind after assignment because operator pointers must target this object.
    bind_operator();

    return *this;
}

template <typename T>
Plane<T>&
Plane<T>::operator=(Plane&& other) noexcept {
    // Avoid self move-assignment.
    if (this == &other) {
        return *this;
    }

    normal = std::move(other.normal);
    offset = other.offset;

    // Rebind both objects so each operator points to its own parameter storage.
    bind_operator();
    other.bind_operator();

    return *this;
}

template <typename T>
void
Plane<T>::bind_operator() noexcept {
    // Store non-owning raw pointers to the plane parameters used by the operator.
    _operator.normal = atlas::raw_pointer_cast(&normal);
    _operator.offset = atlas::raw_pointer_cast(&offset);
}

template <typename T>
typename Plane<T>::Builder
Plane<T>::builder() noexcept {
    // Return a fresh builder for fluent plane construction.
    return Builder {};
}

template <typename T>
GeometryOperator<T>
Plane<T>::make_geometry_operator() const {
    // Wrap the concrete plane operator in the generic geometry operator type.
    return GeometryOperator<T>(_operator);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-point queries to the bound plane operator.
    return _operator.closest_point(p);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate closest-normal queries to the bound plane operator.
    return _operator.closest_normal(p);
}

template <typename T>
T
Plane<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Delegate signed-distance queries to the bound plane operator.
    return _operator.signed_distance(p);
}

template <typename T>
bool
Plane<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate half-space containment checks to the bound plane operator.
    return _operator.is_inside(p, tolerance);
}

template <typename T>
bool
Plane<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Delegate surface-membership checks to the bound plane operator.
    return _operator.is_on_surface(p, tolerance);
}

template <typename T>
atlas::math::Vector<T, 3>
Plane<T>::centroid() const noexcept {
    // Delegate centroid queries to the bound plane operator.
    return _operator.centroid();
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
Plane<T>::bound() const noexcept {
    // Delegate bounding-box construction to the bound plane operator.
    return _operator.bound();
}

template <typename T>
bool
Plane<T>::is_valid() const noexcept {
    // Delegate validity checks to the bound plane operator.
    return _operator.is_valid();
}

template <typename T>
GeometryType
Plane<T>::type() const noexcept {
    // Identify this geometry as a plane.
    return GeometryType::Plane;
}

template <typename T>
Plane<T>
Plane<T>::Builder::build() const {
    // Validate all builder parameters before constructing the final plane.
    validate();

    Plane<T> p {};
    p.normal = _normal;
    p.offset = _offset;

    // Rebind because parameters are assigned after default construction.
    p.bind_operator();

    return p;
}

template <typename T>
atlas::host_shared_ptr<Plane<T>>
Plane<T>::Builder::make_host_shared() const {
    // Build a validated plane and store it in host-managed shared ownership.
    auto p = build();
    return atlas::make_host_shared<Plane<T>>(std::move(p));
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal(const Vector3<T>& normal_) noexcept {
    // Store the requested normal vector for the later build() call.
    _normal = normal_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_offset(T offset_) noexcept {
    // Store the requested implicit-plane offset for the later build() call.
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_normal_offset(const Vector3<T>& normal_, T offset_) noexcept {
    // Store both implicit plane parameters at once.
    _normal = normal_;
    _offset = offset_;
    return *this;
}

template <typename T>
typename Plane<T>::Builder&
Plane<T>::Builder::with_point_normal(const Vector3<T>& point,
                                     const Vector3<T>& normal_) noexcept {
    // Store the normal and derive the offset from the point-normal representation.
    _normal = normal_;
    _offset = -(normal_.dot(point));
    return *this;
}

template <typename T>
void
Plane<T>::Builder::validate() const {
    atlas::geometry::PlaneGeometryOperator<T> op;

    // Validate through the same operator logic used by constructed Plane instances.
    op.normal = atlas::raw_pointer_cast(&_normal);
    op.offset = atlas::raw_pointer_cast(&_offset);

    if (!op.is_valid()) {
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_point(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Without valid plane parameters, there is no meaningful projection target.
    if (!normal || !offset) {
        return p;
    }

    // Signed deviation from the implicit plane equation n dot p + d = 0.
    const T sdev = ((*normal).dot(p) + (*offset));

    // Project the point onto the plane by moving along the plane normal.
    return p - sdev * (*normal);
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::closest_normal(const atlas::math::Vector<T, 3>&) const noexcept {
    // Invalid geometry cannot provide a reliable surface normal.
    if (!normal) {
        return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
    }

    // A plane has the same normal everywhere.
    return *normal;
}

template <typename T>
T
PlaneGeometryOperator<T>::signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept {
    // Invalid geometry is treated as infinitely far away.
    if (!normal || !offset) {
        return std::numeric_limits<T>::infinity();
    }

    // Evaluate the implicit plane equation n dot p + d.
    return (*normal).dot(p) + (*offset);
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_inside(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid geometry cannot contain any point.
    if (!normal || !offset) {
        return false;
    }

    // The inside half-space is the side where signed distance is less than or equal to tolerance.
    return (*normal).dot(p) + (*offset) <= tolerance;
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_on_surface(const atlas::math::Vector<T, 3>& p, const T tolerance) const noexcept {
    // Invalid geometry or negative tolerances cannot accept surface points.
    if (!normal || !offset || tolerance < T(0)) {
        return false;
    }

    const T distance = (*normal).dot(p) + (*offset);
    return distance >= -tolerance && distance <= tolerance;
}

template <typename T>
atlas::math::Vector<T, 3>
PlaneGeometryOperator<T>::centroid() const noexcept {
    // An infinite plane has no finite centroid, so use the origin as a neutral representative point.
    return atlas::math::Vector<T, 3>(T(0), T(0), T(0));
}

template <typename T>
atlas::spatial::AxisAlignedBoundingBox<T>
PlaneGeometryOperator<T>::bound() const noexcept {
    // Represent the infinite plane with the widest finite numeric bounding box.
    const T lo = std::numeric_limits<T>::lowest();
    const T hi = std::numeric_limits<T>::max();

    return atlas::spatial::AxisAlignedBoundingBox<T>(
        atlas::math::Vector<T, 3>(lo, lo, lo),
        atlas::math::Vector<T, 3>(hi, hi, hi));
}

template <typename T>
bool
PlaneGeometryOperator<T>::is_valid() const noexcept {
    // Both normal and offset pointers must be bound before validation can succeed.
    if (!normal || !offset) {
        return false;
    }

    // The normal must be nonzero, and the offset must be finite.
    const T n2 = (*normal).length_squared();
    return atlas::math::isfinite(*normal)
        && (n2 > T(0))
        && atlas::math::isfinite(*offset);
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::trace(const atlas::spatial::Ray<T>& ray) const noexcept {
    HitSurface<T> result {};

    // Invalid geometry produces a default non-intersecting hit result.
    if (!normal || !offset) {
        return result;
    }

    const atlas::math::Vector<T, 3>& n = *normal;
    const T d                          = *offset;

    // Denominator determines whether the ray direction is parallel to the plane.
    const T denom = n.dot(ray.direction);

    // Numerator measures the signed offset from the ray origin to the plane.
    const T numer = -(n.dot(ray.origin) + d);

    if (denom == T(0)) {
        // Parallel rays only intersect if the ray origin already lies on the plane.
        if (numer != T(0)) {
            return result;
        }

        // The entire ray lies on the plane, so report an immediate hit at the origin.
        result.is_intersecting = true;
        result.distance        = T(0);
        result.point           = ray.origin;
        result.normal          = n;

        return result;
    }

    // Solve the ray-plane intersection parameter.
    const T t = numer / denom;

    // Ignore intersections behind the ray origin.
    if (t < T(0)) {
        return result;
    }

    // Populate the hit record with the valid ray-plane intersection.
    result.is_intersecting = true;
    result.distance        = t;
    result.point           = ray.point_at(t);
    result.normal          = n;

    return result;
}

template <typename T>
HitSurface<T>
PlaneGeometryOperator<T>::operator()(const atlas::spatial::Ray<T>& ray) const noexcept {
    // Allow the operator object to be used directly as a ray-intersection functor.
    return trace(ray);
}

} // namespace atlas::geometry
