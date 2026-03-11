#pragma once
namespace atlas::spatial {

template <typename T>
Ray<T>::Ray() noexcept
    : origin(Vector3<T>(T(0), T(0), T(0)))
    , direction(Vector3<T>(T(1), T(0), T(0))) {
    // Default ray:
    // - Origin at the world origin.
    // - Direction along +X axis.
    //
    // This provides a deterministic, valid ray even when default-constructed.
    // (Direction is already unit-length here.)
}

template <typename T>
Ray<T>::Ray(const Vector3<T>& origin_, const Vector3<T>& direction_) noexcept
    : origin(origin_)
    , direction(direction_.normalized()) {
    // Construct a ray from an origin and a direction.
    //
    // Important: the direction is normalized so that:
    // - `t` in point_at(t) corresponds to distance in world units (for typical usage).
    // - Intersection routines can interpret `t` consistently (e.g., comparing hit distances).
    //
    // Note: If `direction` is zero-length, normalized() should handle it safely
    // (implementation-dependent). Callers are expected to pass a non-zero direction.
}

template <typename T>
Vector3<T>
Ray<T>::point_at(T t) const noexcept {
    // Parametric point evaluation:
    //   P(t) = origin + t * direction
    //
    // With a unit-length direction, `t` can be interpreted as "distance traveled"
    // along the ray. Negative `t` values refer to points behind the origin.
    return origin + t * direction;
}

} // namespace atlas::spatial
