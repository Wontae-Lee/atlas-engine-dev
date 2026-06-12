#pragma once
namespace atlas {

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const SurfaceDespawnOperator<T>& op)
    : type(DespawnType::Surface) {
    static_cast<void>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const VolumeDespawnOperator<T>& op)
    : type(DespawnType::Volume) {
    static_cast<void>(op);
}

template <typename T>
DespawnOperator<T>::DespawnOperator(const TracingDespawnOperator<T>& op)
    : type(DespawnType::Tracing) {
    static_cast<void>(op);
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::GeometryOperator<T>& query,
                            const Vector3<T>& vector,
                            const T value) const noexcept {
    switch (type) {
    case DespawnType::Surface:
        return SurfaceDespawnOperator<T>::despawn(query, vector, value);
    case DespawnType::Volume:
        return VolumeDespawnOperator<T>::despawn(query, vector, value);
    case DespawnType::Tracing:
        return TracingDespawnOperator<T>::despawn(query, Vector3<T>(T(0), T(0), T(0)), vector, value);
    default:
        return false;
    }
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::GeometryOperator<T>& query,
                            const Vector3<T>& position,
                            const Vector3<T>& vector,
                            const T value) const noexcept {
    switch (type) {
    case DespawnType::Surface:
        return SurfaceDespawnOperator<T>::despawn(query, vector, value);
    case DespawnType::Volume:
        return VolumeDespawnOperator<T>::despawn(query, vector, value);
    case DespawnType::Tracing:
        return TracingDespawnOperator<T>::despawn(query, position, vector, value);
    default:
        return false;
    }
}

}
