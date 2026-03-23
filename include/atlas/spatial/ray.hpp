#pragma once
namespace atlas::spatial {

template <typename T>
Ray<T>::Ray() noexcept
    : origin(Vector3<T>(T(0), T(0), T(0)))
    , direction(Vector3<T>(T(1), T(0), T(0))) {
}

template <typename T>
Ray<T>::Ray(const Vector3<T>& origin_, const Vector3<T>& direction_) noexcept
    : origin(origin_)
    , direction(direction_.normalized()) {
}

template <typename T>
Vector3<T>
Ray<T>::point_at(T t) const noexcept {

    return origin + t * direction;
}

}