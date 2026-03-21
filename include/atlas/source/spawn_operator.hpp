#pragma once

namespace atlas::system {

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
bool
SurfaceSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                               const Vector3<T>& particle,
                               const T tolerance) const noexcept {
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
bool
VolumeSpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                              const Vector3<T>& particle,
                              const T tolerance) const noexcept {
    return query.is_inside(particle, tolerance);
}

/* ====================================================================== */
/* SpawnOperator special members                                           */
/* ====================================================================== */

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SpawnOperator<T>::SpawnOperator() noexcept
    : type(SpawnType::Surface) {
    new (&surface) SurfaceSpawnOperator<T> {};
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {
    switch (type) {
    case SpawnType::Surface:
        new (&surface) SurfaceSpawnOperator<T> {};
        return;
    case SpawnType::Volume:
        new (&volume) VolumeSpawnOperator<T> {};
        return;
    default:
        this->type = SpawnType::Surface;
        new (&surface) SurfaceSpawnOperator<T> {};
        return;
    }
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SpawnOperator<T>::SpawnOperator(const SpawnOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SpawnOperator<T>&
SpawnOperator<T>::operator=(const SpawnOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SpawnOperator<T>::~SpawnOperator() noexcept {
    destroy_active();
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
SpawnOperator<T>::destroy_active() noexcept {
    switch (type) {
    case SpawnType::Surface:
        surface.~SurfaceSpawnOperator<T>();
        return;
    case SpawnType::Volume:
        volume.~VolumeSpawnOperator<T>();
        return;
    default:
        surface.~SurfaceSpawnOperator<T>();
        return;
    }
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
SpawnOperator<T>::copy_from(const SpawnOperator& other) noexcept {
    switch (type) {
    case SpawnType::Surface:
        new (&surface) SurfaceSpawnOperator<T>(other.surface);
        return;
    case SpawnType::Volume:
        new (&volume) VolumeSpawnOperator<T>(other.volume);
        return;
    default:
        type = SpawnType::Surface;
        new (&surface) SurfaceSpawnOperator<T>(other.surface);
        return;
    }
}

/* ====================================================================== */
/* SpawnOperator tagged constructors                                       */
/* ====================================================================== */

template <typename T>
ATLAS_HOST
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {
    new (&surface) SurfaceSpawnOperator<T>(op);
}

template <typename T>
ATLAS_HOST
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {
    new (&volume) VolumeSpawnOperator<T>(op);
}

/* ====================================================================== */
/* Spawn dispatch                                                          */
/* ====================================================================== */

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
bool
SpawnOperator<T>::spawn(const atlas::geometry::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {
    switch (type) {
    case SpawnType::Surface:
        return surface.spawn(query, particle, tolerance);
    case SpawnType::Volume:
        return volume.spawn(query, particle, tolerance);
    default:
        return false;
    }
}

} // namespace atlas::system
