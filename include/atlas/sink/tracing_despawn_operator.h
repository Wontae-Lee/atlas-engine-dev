#pragma once

#include <atlas/geometry/geometry_operator.h>
#include <atlas/spatial/ray.h>

namespace atlas {

template <typename T>
struct TracingDespawnOperator final {

    ATLAS_ALL_DEVICE ATLAS_NODISCARD static ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& position,
            const Vector3<T>& velocity,
            T time = T(0)) noexcept;
};

}

#include <atlas/sink/tracing_despawn_operator.hpp>