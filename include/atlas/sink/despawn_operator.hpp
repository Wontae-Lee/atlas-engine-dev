#pragma once

namespace atlas::system {

template <typename T>
bool
SurfaceDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                   const Vector3<T>& particle,
                                   const T tolerance) noexcept {
    // Despawn particle only when it lies on the geometry surface.
    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeDespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                                  const Vector3<T>& particle,
                                  const T tolerance) noexcept {
    // Despawn particle only when it lies inside the geometry volume.
    return query.is_inside(particle, tolerance);
}

template <typename T>
DespawnOperator<T>::DespawnOperator() noexcept
    : type(DespawnType::Surface) {
    // Default-construct as Surface despawn operator.
    // Placement-new is used to initialize the active union member.
    new (&surface) SurfaceDespawnOperator<T> {};
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {

    // Construct the active union member according to the requested despawn type.
    switch (type) {
    case DespawnType::Surface:
        // Initialize as surface-based despawn rule.
        new (&surface) SurfaceDespawnOperator<T> {};
        return;

    case DespawnType::Volume:
        // Initialize as volume-based despawn rule.
        new (&volume) VolumeDespawnOperator<T> {};
        return;

    default:
        // Fallback to Surface so the operator always remains in a valid state.
        this->type = DespawnType::Surface;
        new (&surface) SurfaceDespawnOperator<T> {};
        return;
    }
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnOperator& other) noexcept
    : type(other.type) {
    // Copy-construct the currently active union member from the source object.
    copy_from(other);
}

template <typename T>
DespawnOperator<T>&
DespawnOperator<T>::operator=(const DespawnOperator& other) noexcept {

    // Guard against self-assignment.
    if (this == &other) return *this;

    // Destroy the currently active union member before reconstructing it.
    destroy_active();

    // Copy the active type tag from the source object.
    type = other.type;

    // Reconstruct the matching union member from the source object.
    copy_from(other);

    return *this;
}

template <typename T>
DespawnOperator<T>::~DespawnOperator() noexcept {
    // Destroy the currently active union member on object teardown.
    destroy_active();
}

template <typename T>
void
DespawnOperator<T>::destroy_active() noexcept {

    // Explicitly destroy the union member indicated by the active type tag.
    switch (type) {
    case DespawnType::Surface:
        // Active member is `surface`.
        surface.~SurfaceDespawnOperator<T>();
        return;

    case DespawnType::Volume:
        // Active member is `volume`.
        volume.~VolumeDespawnOperator<T>();
        return;

    default:
        // Conservative fallback: destroy as surface operator.
        surface.~SurfaceDespawnOperator<T>();
        return;
    }
}

template <typename T>
void
DespawnOperator<T>::copy_from(const DespawnOperator& other) noexcept {

    // Copy-construct the correct union member using placement-new.
    switch (type) {
    case DespawnType::Surface:
        // Copy the surface despawn operator from the source object.
        new (&surface) SurfaceDespawnOperator<T>(other.surface);
        return;

    case DespawnType::Volume:
        // Copy the volume despawn operator from the source object.
        new (&volume) VolumeDespawnOperator<T>(other.volume);
        return;

    default:
        // Fallback to Surface to preserve a valid active state.
        type = DespawnType::Surface;
        new (&surface) SurfaceDespawnOperator<T>(other.surface);
        return;
    }
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {
    // Construct directly from a SurfaceDespawnOperator instance.
    new (&surface) SurfaceDespawnOperator<T>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {
    // Construct directly from a VolumeDespawnOperator instance.
    new (&volume) VolumeDespawnOperator<T>(op);
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::geometry::GeometryOperator<T>& query,
                            const Vector3<T>& particle,
                            const T tolerance) const noexcept {

    // Dispatch despawn behavior to the active operator implementation.
    switch (type) {
    case DespawnType::Surface:
        // Use surface-based despawn test.
        return surface.despawn(query, particle, tolerance);

    case DespawnType::Volume:
        // Use volume-based despawn test.
        return volume.despawn(query, particle, tolerance);

    default:
        // Invalid / unknown state: do not despawn.
        return false;
    }
}

} // namespace atlas::system