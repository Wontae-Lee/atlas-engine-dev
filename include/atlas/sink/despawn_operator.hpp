#pragma once
namespace atlas::fluid {
template <typename T>
bool
SurfaceDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) noexcept {
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) noexcept {
    return query.is_inside(particle, tolerance);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {
    static_cast<void>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {
    static_cast<void>(op);
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                            const Vector3<T>& particle,
                            const T tolerance) const noexcept {
    switch (type) {
    case DespawnType::Surface:
        return SurfaceDespawnOperator<T>::despawn(query, particle, tolerance);
    case DespawnType::Volume:
        return VolumeDespawnOperator<T>::despawn(query, particle, tolerance);
    default:
        return false;
    }
}

}