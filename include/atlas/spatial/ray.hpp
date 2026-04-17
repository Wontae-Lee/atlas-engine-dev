#pragma once
namespace atlas::spatial {

template <typename T>
Ray<T>::Ray() noexcept
    : origin(Vector3<T>(T(0), T(0), T(0)))
    , direction(Vector3<T>(T(1), T(0), T(0))) {
    // Construct a default ray.
    //
    // Default convention:
    // - origin    = (0, 0, 0)
    // - direction = (1, 0, 0)
    //
    // This gives a valid canonical ray that starts at the world origin
    // and points along the positive x-axis.
}

template <typename T>
Ray<T>::Ray(const Vector3<T>& origin_, const Vector3<T>& direction_) noexcept
    : origin(origin_)
    , direction(direction_.normalized()) {
    // Construct a ray from an explicit origin and direction.
    //
    // Important design choice:
    // - The input direction is normalized on construction.
    //
    // Consequences:
    // - `direction` stores a unit vector whenever the input direction is valid.
    // - Parametric distance `t` in point_at(t) is then interpreted directly
    //   in world-space length units.
    //
    // Note:
    // - This relies on Vector3<T>::normalized() handling degenerate input
    //   consistently if a zero-length direction is ever provided.
}

template <typename T>
Vector3<T>
Ray<T>::point_at(T t) const noexcept {
    // Evaluate the ray at parameter t.
    //
    // Parametric ray equation:
    //   p(t) = origin + t * direction
    //
    // Interpretation:
    // - t = 0   -> origin
    // - t > 0   -> points in the forward ray direction
    // - t < 0   -> points behind the ray origin along the same line
    //
    // Because direction is normalized by the non-default constructor,
    // `t` usually corresponds to geometric distance along the ray.
    return origin + t * direction;
}

} // namespace atlas::spatial