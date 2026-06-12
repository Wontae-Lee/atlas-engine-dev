#pragma once

namespace atlas {

template <typename T>
bool
VolumeDespawnOperator<T>::despawn(const atlas::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) noexcept {
    return query.is_inside(particle, tolerance);
}

}