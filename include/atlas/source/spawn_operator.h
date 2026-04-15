#pragma once

#include <atlas/math/math.h>

namespace atlas::fluid {

enum class SpawnType : int {
    Surface,
    Volume
};

template <typename T>
struct SurfaceSpawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

template <typename T>
struct VolumeSpawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) noexcept;
};

template <typename T>
struct SpawnOperator final {

    SpawnType type = SpawnType::Surface;

    union {
        SurfaceSpawnOperator<T> surface;
        VolumeSpawnOperator<T> volume;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit SpawnOperator(SpawnType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator(const SpawnOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SpawnOperator&
    operator=(const SpawnOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SpawnOperator() noexcept;

    ATLAS_HOST
    SpawnOperator(const SurfaceSpawnOperator<T>& op);

    ATLAS_HOST
    SpawnOperator(const VolumeSpawnOperator<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SpawnOperator& other) noexcept;
};

}

#include <atlas/source/spawn_operator.hpp>