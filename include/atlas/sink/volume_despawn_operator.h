#pragma once

#include <atlas/geometry/geometry_operator.h>

namespace atlas {

template <typename T>
struct VolumeDespawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) noexcept;
};

}

#include <atlas/sink/volume_despawn_operator.hpp>