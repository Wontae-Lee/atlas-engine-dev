#pragma once

#include <atlas/math/math.h>

namespace atlas::fluid {

enum class DespawnType : int {

    Surface,

    Volume
};

template <typename T>
struct SurfaceDespawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

template <typename T>
struct VolumeDespawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

template <typename T>
struct DespawnOperator final {

    DespawnType type = DespawnType::Surface;

    union {

        SurfaceDespawnOperator<T> surface;

        VolumeDespawnOperator<T> volume;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DespawnOperator(DespawnType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator(const DespawnOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DespawnOperator&
    operator=(const DespawnOperator& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DespawnOperator() noexcept;

    ATLAS_HOST
    DespawnOperator(const SurfaceDespawnOperator<T>& op);

    ATLAS_HOST
    DespawnOperator(const VolumeDespawnOperator<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DespawnOperator& other) noexcept;
};

}

#include <atlas/sink/despawn_operator.hpp>