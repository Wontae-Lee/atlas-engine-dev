#pragma once

namespace atlas::fluid {

template <typename T>
bool
SurfaceSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                               const Vector3<T>& particle,
                               const T tolerance) noexcept {

    // Accept the particle only if it lies on the queried geometry surface
    // within the requested tolerance.
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                              const Vector3<T>& particle,
                              const T tolerance) noexcept {

    // Accept the particle only if it lies inside the queried geometry region
    // within the requested tolerance.
    return query.is_inside(particle, tolerance);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {

    // The policy is stateless, so constructing from it only selects the tag.
    static_cast<void>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {

    // The policy is stateless, so constructing from it only selects the tag.
    static_cast<void>(op);
}

template <typename T>
bool
SpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {

    // Dispatch the spawn decision to the currently active concrete policy.
    switch (type) {
    case SpawnType::Surface:
        return SurfaceSpawnOperator<T>::spawn(query, particle, tolerance);

    case SpawnType::Volume:
        return VolumeSpawnOperator<T>::spawn(query, particle, tolerance);

    default:

        // Defensive fallback for an invalid runtime tag.
        return false;
    }
}

}
