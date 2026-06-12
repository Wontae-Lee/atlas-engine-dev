#pragma once

namespace atlas {

template <typename T>
bool
SurfaceSpawnOperator<T>::spawn(const atlas::GeometryOperator<T>& query,
                               const Vector3<T>& particle,
                               const T tolerance) noexcept {

    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeSpawnOperator<T>::spawn(const atlas::GeometryOperator<T>& query,
                              const Vector3<T>& particle,
                              const T tolerance) noexcept {

    return query.is_inside(particle, tolerance);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {

    static_cast<void>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {

    static_cast<void>(op);
}

template <typename T>
bool
SpawnOperator<T>::spawn(const atlas::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {

    switch (type) {
    case SpawnType::Surface:
        return SurfaceSpawnOperator<T>::spawn(query, particle, tolerance);

    case SpawnType::Volume:
        return VolumeSpawnOperator<T>::spawn(query, particle, tolerance);

    default:
        return false;
    }
}

}