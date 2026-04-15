#pragma once

namespace atlas::system {

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
DespawnOperator<T>::DespawnOperator() noexcept
    : type(DespawnType::Surface) {

    new (&surface) SurfaceDespawnOperator<T> {};
}

template <typename T>
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
DespawnOperator<T>::DespawnOperator(const DespawnOperator& other) noexcept
    : type(other.type) {

    copy_from(other);
}

template <typename T>
DespawnOperator<T>&
DespawnOperator<T>::operator=(const DespawnOperator& other) noexcept {

    if (this == &other) return *this;

    destroy_active();

    type = other.type;

    copy_from(other);

    return *this;
}

template <typename T>
DespawnOperator<T>::~DespawnOperator() noexcept {

    destroy_active();
}

template <typename T>
void
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
void
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

template <typename T>
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {

    new (&surface) SurfaceDespawnOperator<T>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {

    new (&volume) VolumeDespawnOperator<T>(op);
}

template <typename T>
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

}