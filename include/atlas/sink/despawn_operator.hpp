#pragma once

namespace atlas::fluid {

template <typename T>
bool
SurfaceDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) noexcept {

    // Remove the particle only if it lies on the queried geometry surface
    // within the requested tolerance.
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) noexcept {

    // Remove the particle only if it lies inside the queried geometry region
    // within the requested tolerance.
    return query.is_inside(particle, tolerance);
}

template <typename T>
DespawnOperator<T>::DespawnOperator() noexcept
    : type(DespawnType::Surface) {

    // Default-initialize the tagged union with a surface despawn policy.
    new (&surface) SurfaceDespawnOperator<T> {};
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {

    // Construct the union member corresponding to the requested despawn type.
    switch (type) {
    case DespawnType::Surface:

        new (&surface) SurfaceDespawnOperator<T> {};
        return;

    case DespawnType::Volume:

        new (&volume) VolumeDespawnOperator<T> {};
        return;

    default:

        // Defensive fallback to a valid surface-based state.
        this->type = DespawnType::Surface;
        new (&surface) SurfaceDespawnOperator<T> {};
        return;
    }
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnOperator& other) noexcept
    : type(other.type) {

    // Copy the runtime tag first, then reconstruct the matching active member.
    copy_from(other);
}

template <typename T>
DespawnOperator<T>&
DespawnOperator<T>::operator=(const DespawnOperator& other) noexcept {

    // Self-assignment requires no work.
    if (this == &other) return *this;

    // Since this type manually manages a tagged union, assignment must:
    // 1. destroy the current active member
    // 2. copy the runtime tag
    // 3. reconstruct the new active member
    destroy_active();

    type = other.type;

    copy_from(other);

    return *this;
}

template <typename T>
DespawnOperator<T>::~DespawnOperator() noexcept {

    // Explicitly destroy the currently active union member.
    destroy_active();
}

template <typename T>
void
DespawnOperator<T>::destroy_active() noexcept {

    // Destroy the concrete despawn policy selected by the runtime tag.
    switch (type) {
    case DespawnType::Surface:

        surface.~SurfaceDespawnOperator<T>();
        return;

    case DespawnType::Volume:

        volume.~VolumeDespawnOperator<T>();
        return;

    default:

        // Defensive fallback.
        surface.~SurfaceDespawnOperator<T>();
        return;
    }
}

template <typename T>
void
DespawnOperator<T>::copy_from(const DespawnOperator& other) noexcept {

    // Construct the active union member from the corresponding member in `other`.
    //
    // This function assumes:
    // - `type` is already set correctly
    // - any previously active member has already been destroyed
    switch (type) {
    case DespawnType::Surface:

        new (&surface) SurfaceDespawnOperator<T>(other.surface);
        return;

    case DespawnType::Volume:

        new (&volume) VolumeDespawnOperator<T>(other.volume);
        return;

    default:

        // Defensive fallback to a valid surface-based state.
        type = DespawnType::Surface;
        new (&surface) SurfaceDespawnOperator<T>(other.surface);
        return;
    }
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {

    // Construct this tagged union directly from a surface despawn policy.
    new (&surface) SurfaceDespawnOperator<T>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {

    // Construct this tagged union directly from a volume despawn policy.
    new (&volume) VolumeDespawnOperator<T>(op);
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                            const Vector3<T>& particle,
                            const T tolerance) const noexcept {

    // Dispatch the despawn decision to the currently active concrete policy.
    switch (type) {
    case DespawnType::Surface:

        return surface.despawn(query, particle, tolerance);

    case DespawnType::Volume:

        return volume.despawn(query, particle, tolerance);

    default:

        // Defensive fallback for an invalid runtime tag.
        return false;
    }
}

} // namespace atlas::fluid