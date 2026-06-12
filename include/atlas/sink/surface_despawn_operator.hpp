#pragma once

namespace atlas {

template <typename T>
bool
SurfaceDespawnOperator<T>::despawn(const atlas::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) noexcept {
    return query.is_on_surface(particle, tolerance);
}

}