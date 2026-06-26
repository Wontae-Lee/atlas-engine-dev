#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/sink/surface_despawn_operator.h>
#include <atlas/sink/tracing_despawn_operator.h>
#include <atlas/sink/volume_despawn_operator.h>

namespace atlas {

enum class DespawnType : int {

    Surface,

    Volume,

    Tracing
};

template <typename T>
struct DespawnOperator final {

    DespawnType type = DespawnType::Surface;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit DespawnOperator(DespawnType type) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator(const DespawnOperator& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DespawnOperator&
    operator=(const DespawnOperator& other) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~DespawnOperator() noexcept = default;

    ATLAS_HOST
    DespawnOperator(const SurfaceDespawnOperator<T>& op);

    ATLAS_HOST
    DespawnOperator(const VolumeDespawnOperator<T>& op);

    ATLAS_HOST
    DespawnOperator(const TracingDespawnOperator<T>& op);

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& vector,
            T value = T(0)) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const atlas::GeometryOperator<T>& query,
            const Vector3<T>& position,
            const Vector3<T>& vector,
            T value = T(0)) const noexcept;
};

}

#include <atlas/sink/despawn_operator.hpp>