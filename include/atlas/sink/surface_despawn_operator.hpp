#pragma once

namespace atlas::fluid {

template <typename T>
bool
SurfaceDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) noexcept {
    return query.is_on_surface(particle, tolerance);
}

} // namespace atlas::fluid
