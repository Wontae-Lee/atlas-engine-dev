#pragma once

namespace atlas::fluid {

template <typename T>
bool
SurfaceDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) noexcept {

    // Remove the particle only when it lies on the queried geometry surface
    // within the provided tolerance.
    //
    // This policy is appropriate for sink-like behavior defined strictly on
    // boundary surfaces rather than on enclosed volume membership.
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) noexcept {

    // Remove the particle only when it lies inside the queried geometry region
    // within the provided tolerance.
    //
    // This policy is appropriate for sinks or removal zones defined by an
    // enclosed region rather than a boundary shell alone.
    return query.is_inside(particle, tolerance);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {

    // Store the runtime dispatch tag directly.
    //
    // The operator itself holds no additional state because all supported
    // concrete despawn policies are stateless.
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {

    // Constructing from a stateless surface policy only selects the runtime tag.
    // No additional payload needs to be copied or stored.
    static_cast<void>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {

    // Constructing from a stateless volume policy only selects the runtime tag.
    // No additional payload needs to be copied or stored.
    static_cast<void>(op);
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                            const Vector3<T>& particle,
                            const T tolerance) const noexcept {

    // Dispatch the despawn decision to the active concrete policy.
    //
    // Because the supported policies are stateless, runtime dispatch only
    // depends on the stored enum tag.
    switch (type) {
    case DespawnType::Surface:
        return SurfaceDespawnOperator<T>::despawn(query, particle, tolerance);

    case DespawnType::Volume:
        return VolumeDespawnOperator<T>::despawn(query, particle, tolerance);

    default:

        // Defensive fallback for an invalid or corrupted runtime tag.
        //
        // Returning false is the safest behavior here because it prevents
        // accidental particle removal when the dispatch state is invalid.
        return false;
    }
}

} // namespace atlas::fluid