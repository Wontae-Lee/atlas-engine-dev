#pragma once

namespace atlas::fluid {

template <typename T>
bool
TracingDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& position,
                                   const Vector3<T>& velocity,
                                   const T time) noexcept {
    const T speed = velocity.length();
    if (!(time > T(0)) || !(speed > T(0))) {
        return false;
    }
    const auto hit = query.trace(atlas::spatial::Ray<T>(position, velocity));
    return hit.is_intersecting && hit.distance >= T(0) && hit.distance <= speed * time;
}

} // namespace atlas::fluid
