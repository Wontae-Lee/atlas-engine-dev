#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/math/math.h>

#include <type_traits>

namespace atlas {

enum class SpawnType : int {

    Surface,

    Volume
};

template <typename T>
struct SurfaceSpawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

template <typename T>
struct VolumeSpawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

namespace detail {

    template <typename T>
    using SpawnTypeSwitch = DeviceTypeSwitch<
        SpawnType,
        SpawnType::Surface,
        DeviceTypeCase<SpawnType, SpawnType::Surface, SurfaceSpawnOperator<T>>,
        DeviceTypeCase<SpawnType, SpawnType::Volume, VolumeSpawnOperator<T>>>;

}

template <typename T>
struct SpawnOperator final {

    SpawnType type = SpawnType::Surface;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SpawnOperator(SpawnType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator(const SpawnOperator& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SpawnOperator&
    operator=(const SpawnOperator& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SpawnOperator() noexcept = default;

    template <typename Payload,
              std::enable_if_t<detail::SpawnTypeSwitch<T>::template holds<std::decay_t<Payload>>, int> = 0>
    ATLAS_HOST SpawnOperator(const Payload&) noexcept
        : type(detail::SpawnTypeSwitch<T>::template tag_of<std::decay_t<Payload>>()) {
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    spawn(const atlas::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) const noexcept;
};

}

#include <atlas/source/spawn_operator.hpp>