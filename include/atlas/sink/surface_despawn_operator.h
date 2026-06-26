#pragma once

#include <atlas/geometry/geometry_operator.h>

namespace atlas {

template <typename T>
struct SurfaceDespawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>&,
            const Vector3<T>& particle,
            T tolerance) noexcept {
        return despawn(query, particle, tolerance);
    }
};

}

#include <atlas/sink/surface_despawn_operator.hpp>