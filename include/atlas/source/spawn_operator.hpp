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
SpawnOperator<T>::SpawnOperator() noexcept
    : type(SpawnType::Surface) {

    // Default-initialize the tagged union with a surface spawn policy.
    new (&surface) SurfaceSpawnOperator<T> {};
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {

    // Construct the union member corresponding to the requested spawn type.
    switch (type) {
    case SpawnType::Surface:
        new (&surface) SurfaceSpawnOperator<T> {};
        return;

    case SpawnType::Volume:
        new (&volume) VolumeSpawnOperator<T> {};
        return;

    default:

        // Defensive fallback to a valid surface-based state.
        this->type = SpawnType::Surface;
        new (&surface) SurfaceSpawnOperator<T> {};
        return;
    }
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnOperator& other) noexcept
    : type(other.type) {

    // Copy the runtime tag first, then reconstruct the matching active member.
    copy_from(other);
}

template <typename T>
SpawnOperator<T>&
SpawnOperator<T>::operator=(const SpawnOperator& other) noexcept {

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
SpawnOperator<T>::~SpawnOperator() noexcept {

    // Explicitly destroy the currently active union member.
    destroy_active();
}

template <typename T>
void
SpawnOperator<T>::destroy_active() noexcept {

    // Destroy the concrete spawn policy selected by the runtime tag.
    switch (type) {
    case SpawnType::Surface:
        surface.~SurfaceSpawnOperator<T>();
        return;

    case SpawnType::Volume:
        volume.~VolumeSpawnOperator<T>();
        return;

    default:

        // Defensive fallback.
        surface.~SurfaceSpawnOperator<T>();
        return;
    }
}

template <typename T>
void
SpawnOperator<T>::copy_from(const SpawnOperator& other) noexcept {

    // Construct the active union member from the corresponding member in `other`.
    //
    // This function assumes:
    // - `type` is already set correctly
    // - any previously active member has already been destroyed
    switch (type) {
    case SpawnType::Surface:
        new (&surface) SurfaceSpawnOperator<T>(other.surface);
        return;

    case SpawnType::Volume:
        new (&volume) VolumeSpawnOperator<T>(other.volume);
        return;

    default:

        // Defensive fallback to a valid surface-based state.
        type = SpawnType::Surface;
        new (&surface) SurfaceSpawnOperator<T>(other.surface);
        return;
    }
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {

    // Construct this tagged union directly from a surface spawn policy.
    new (&surface) SurfaceSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {

    // Construct this tagged union directly from a volume spawn policy.
    new (&volume) VolumeSpawnOperator<T>(op);
}

template <typename T>
bool
SpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {

    // Dispatch the spawn decision to the currently active concrete policy.
    switch (type) {
    case SpawnType::Surface:
        return surface.spawn(query, particle, tolerance);

    case SpawnType::Volume:
        return volume.spawn(query, particle, tolerance);

    default:

        // Defensive fallback for an invalid runtime tag.
        return false;
    }
}

}