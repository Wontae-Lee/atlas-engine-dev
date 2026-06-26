#pragma once

namespace atlas {

namespace detail {

    template <typename T>
    using SpawnTypeSwitch = DeviceTypeSwitch<
        SpawnType,
        SpawnType::Surface,
        DeviceTypeCase<SpawnType, SpawnType::Surface, SurfaceSpawnOperator<T>>,
        DeviceTypeCase<SpawnType, SpawnType::Volume, VolumeSpawnOperator<T>>>;

}

template <typename T>
bool
SurfaceSpawnOperator<T>::spawn(const atlas::GeometryOperator<T>& query,
                               const Vector3<T>& particle,
                               const T tolerance) noexcept {

    return query.is_on_surface(particle, tolerance);
}

template <typename T>
bool
VolumeSpawnOperator<T>::spawn(const atlas::GeometryOperator<T>& query,
                              const Vector3<T>& particle,
                              const T tolerance) noexcept {

    return query.is_inside(particle, tolerance);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type) noexcept
    : type(type) {
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const SurfaceSpawnOperator<T>& op)
    : type(SpawnType::Surface) {

    static_cast<void>(op);
}

template <typename T>
SpawnOperator<T>::SpawnOperator(const VolumeSpawnOperator<T>& op)
    : type(SpawnType::Volume) {

    static_cast<void>(op);
}

template <typename T>
bool
SpawnOperator<T>::spawn(const atlas::GeometryOperator<T>& query,
                        const Vector3<T>& particle,
                        const T tolerance) const noexcept {

    return detail::SpawnTypeSwitch<T>::visit(
        type,
        [&] ATLAS_ALL_DEVICE (auto spawn_tag) noexcept {
            using Op = typename decltype(spawn_tag)::type;
            return Op::spawn(query, particle, tolerance);
        },
        false);
}

}