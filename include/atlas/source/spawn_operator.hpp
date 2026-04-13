#pragma once

namespace atlas::system {

template <typename T>
bool
SurfaceSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                               const Vector3<T>& particle,
                               const T tolerance) noexcept {
    // Accept particle only if it lies on the surface of the geometry.
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                              const Vector3<T>& particle,
                              const T tolerance) noexcept {
    // Accept particle only if it lies inside the geometry volume.
    return query.is_inside(particle, tolerance);
}

template <typename T>
SpawnOperator<T>::SpawnOperator() noexcept
    : type(SpawnType::Surface) {
    // Default construct as Surface operator.
    // Placement-new is used to construct the active union member.
    new (&surface) SurfaceSpawnOperator<T> {};
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {

    // Construct the appropriate operator in-place depending on SpawnType.
    switch (type) {
    case SpawnType::Surface:
        new (&surface) SurfaceSpawnOperator<T> {};
        return;

    case SpawnType::Volume:
        new (&volume) VolumeSpawnOperator<T> {};
        return;

    default:
        // Fallback to Surface to guarantee a valid state.
        this->type = SpawnType::Surface;
        new (&surface) SurfaceSpawnOperator<T> {};
        return;
    }
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnOperator& other) noexcept
    : type(other.type) {

    // Copy-construct active union member from other.
    copy_from(other);
}

template <typename T>
SpawnOperator<T>&
SpawnOperator<T>::operator=(const SpawnOperator& other) noexcept {

    // Self-assignment guard.
    if (this == &other) return *this;

    // Destroy current active union member before overwriting.
    destroy_active();

    // Copy type and reconstruct active member.
    type = other.type;
    copy_from(other);

    return *this;
}

template <typename T>
SpawnOperator<T>::~SpawnOperator() noexcept {

    // Destroy currently active union member.
    destroy_active();
}

template <typename T>
void
SpawnOperator<T>::destroy_active() noexcept {

    // Explicitly call destructor of the active union member.
    switch (type) {
    case SpawnType::Surface:
        surface.~SurfaceSpawnOperator<T>();
        return;

    case SpawnType::Volume:
        volume.~VolumeSpawnOperator<T>();
        return;

    default:
        // Fallback: assume surface for safety.
        surface.~SurfaceSpawnOperator<T>();
        return;
    }
}

template <typename T>
void
SpawnOperator<T>::copy_from(const SpawnOperator& other) noexcept {

    // Copy-construct the correct union member using placement-new.
    switch (type) {
    case SpawnType::Surface:
        new (&surface) SurfaceSpawnOperator<T>(other.surface);
        return;

    case SpawnType::Volume:
        new (&volume) VolumeSpawnOperator<T>(other.volume);
        return;

    default:
        // Fallback to Surface to maintain invariant.
        type = SpawnType::Surface;
        new (&surface) SurfaceSpawnOperator<T>(other.surface);
        return;
    }
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {

    // Construct Surface operator directly from given instance.
    new (&surface) SurfaceSpawnOperator<T>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {

    // Construct Volume operator directly from given instance.
    new (&volume) VolumeSpawnOperator<T>(op);
}

template <typename T>
bool
SpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {

    // Dispatch spawn behavior based on active operator type.
    switch (type) {
    case SpawnType::Surface:
        return surface.spawn(query, particle, tolerance);

    case SpawnType::Volume:
        return volume.spawn(query, particle, tolerance);

    default:
        // Invalid state: reject particle.
        return false;
    }
}

}