#pragma once
namespace atlas {

template <typename T>
DespawnOperator<T>::DespawnOperator(const DespawnType type) noexcept
    : type(type) {
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