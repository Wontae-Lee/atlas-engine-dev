#pragma once

namespace atlas {

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