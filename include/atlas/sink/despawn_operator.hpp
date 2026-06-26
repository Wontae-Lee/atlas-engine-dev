#pragma once
namespace atlas {

namespace detail {

    template <typename T>
    using DespawnTypeSwitch = DeviceTypeSwitch<
        DespawnType,
        DespawnType::Surface,
        DeviceTypeCase<DespawnType, DespawnType::Surface, SurfaceDespawnOperator<T>>,
        DeviceTypeCase<DespawnType, DespawnType::Volume, VolumeDespawnOperator<T>>,
        DeviceTypeCase<DespawnType, DespawnType::Tracing, TracingDespawnOperator<T>>>;

}

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
    return detail::DespawnTypeSwitch<T>::visit(
        type,
        [&] ATLAS_ALL_DEVICE (auto despawn_tag) noexcept {
            using Op = typename decltype(despawn_tag)::type;
            return Op::despawn(query, Vector3<T>(T(0), T(0), T(0)), vector, value);
        },
        false);
}

template <typename T>
bool
DespawnOperator<T>::despawn(const atlas::GeometryOperator<T>& query,
                            const Vector3<T>& position,
                            const Vector3<T>& vector,
                            const T value) const noexcept {
    return detail::DespawnTypeSwitch<T>::visit(
        type,
        [&] ATLAS_ALL_DEVICE (auto despawn_tag) noexcept {
            using Op = typename decltype(despawn_tag)::type;
            return Op::despawn(query, position, vector, value);
        },
        false);
}

}