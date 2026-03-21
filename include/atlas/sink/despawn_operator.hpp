#pragma once

namespace atlas::system {

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
bool
SurfaceDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) const noexcept {
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
bool
VolumeDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) const noexcept {
    return query.is_inside(particle, tolerance);
}

/* ====================================================================== */
/* DespawnOperator special members                                         */
/* ====================================================================== */

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DespawnOperator<T>::DespawnOperator() noexcept
    : type(DespawnType::Surface) {
    new (&surface) SurfaceDespawnOperator<T> {};
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {
    switch (type) {
    case DespawnType::Surface:
        new (&surface) SurfaceDespawnOperator<T> {};
        return;
    case DespawnType::Volume:
        new (&volume) VolumeDespawnOperator<T> {};
        return;
    default:
        this->type = DespawnType::Surface;
        new (&surface) SurfaceDespawnOperator<T> {};
        return;
    }
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DespawnOperator<T>::DespawnOperator(const DespawnOperator& other) noexcept
    : type(other.type) {
    copy_from(other);
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DespawnOperator<T>&
DespawnOperator<T>::operator=(const DespawnOperator& other) noexcept {
    if (this == &other) return *this;
    destroy_active();
    type = other.type;
    copy_from(other);
    return *this;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DespawnOperator<T>::~DespawnOperator() noexcept {
    destroy_active();
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DespawnOperator<T>::destroy_active() noexcept {
    switch (type) {
    case DespawnType::Surface:
        surface.~SurfaceDespawnOperator<T>();
        return;
    case DespawnType::Volume:
        volume.~VolumeDespawnOperator<T>();
        return;
    default:
        surface.~SurfaceDespawnOperator<T>();
        return;
    }
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DespawnOperator<T>::copy_from(const DespawnOperator& other) noexcept {
    switch (type) {
    case DespawnType::Surface:
        new (&surface) SurfaceDespawnOperator<T>(other.surface);
        return;
    case DespawnType::Volume:
        new (&volume) VolumeDespawnOperator<T>(other.volume);
        return;
    default:
        type = DespawnType::Surface;
        new (&surface) SurfaceDespawnOperator<T>(other.surface);
        return;
    }
}

/* ====================================================================== */
/* DespawnOperator tagged constructors                                     */
/* ====================================================================== */

template <typename T>
ATLAS_HOST
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {
    new (&surface) SurfaceDespawnOperator<T>(op);
}

template <typename T>
ATLAS_HOST
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {
    new (&volume) VolumeDespawnOperator<T>(op);
}

/* ====================================================================== */
/* Despawn dispatch                                                        */
/* ====================================================================== */

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
bool
DespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                            const Vector3<T>& particle,
                            const T tolerance) const noexcept {
    switch (type) {
    case DespawnType::Surface:
        return surface.despawn(query, particle, tolerance);
    case DespawnType::Volume:
        return volume.despawn(query, particle, tolerance);
    default:
        return false;
    }
}

} // namespace atlas::system
