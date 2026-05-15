#pragma once

namespace atlas::fluid {

template <typename T>
bool
VolumeDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) noexcept {
    return query.is_inside(particle, tolerance);
}

} // namespace atlas::fluid
