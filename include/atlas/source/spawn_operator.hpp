#pragma once

namespace atlas::fluid {

template <typename T>
bool
SurfaceSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                               const Vector3<T>& particle,
                               const T tolerance) noexcept {
    // Accept particles that lie on the queried geometry surface.
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                              const Vector3<T>& particle,
                              const T tolerance) noexcept {
    // Accept particles that lie inside the queried geometry volume.
    return query.is_inside(particle, tolerance);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {
    // The concrete operator carries no state; only the type tag is stored.
    static_cast<void>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {
    // The concrete operator carries no state; only the type tag is stored.
    static_cast<void>(op);
}

template <typename T>
bool
SpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {
    // Dispatch the spawn test based on the stored spawn mode.
    switch (type) {
    case SpawnType::Surface:
        return SurfaceSpawnOperator<T>::spawn(query, particle, tolerance);

    case SpawnType::Volume:
        return VolumeSpawnOperator<T>::spawn(query, particle, tolerance);

    default:
        return false;
    }
}

} // namespace atlas::fluid