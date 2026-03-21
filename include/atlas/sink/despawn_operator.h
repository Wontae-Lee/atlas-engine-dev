#pragma once

#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Despawn classification mode used by @ref DespawnOperator.
 *
 * @tparam T Floating-point scalar type.
 */
enum class DespawnType : int {
    Surface,
    Volume
};

/**
 * @brief Surface despawn predicate for a single particle position.
 *
 * @details
 * Returns `true` when the particle should be despawned by surface classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceDespawnOperator final {
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const atlas::geometry::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) const noexcept;
};

/**
 * @brief Volume despawn predicate for a single particle position.
 *
 * @details
 * Returns `true` when the particle should be despawned by volume classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct VolumeDespawnOperator final {
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    despawn(const atlas::geometry::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) const noexcept;
};

/**
 * @brief Runtime-dispatched despawn operator for surface or volume classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct DespawnOperator final {
    DespawnType type = DespawnType::Surface;
    union {
        SurfaceDespawnOperator<T> surface;
        VolumeDespawnOperator<T> volume;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    DespawnOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    explicit DespawnOperator(DespawnType type) noexcept;

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
    despawn(const atlas::geometry::GeometryOperator<T>& query,
            const Vector3<T>& particle,
            T tolerance = T(0)) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const DespawnOperator& other) noexcept;
};

} // namespace atlas::system

#include <atlas/sink/despawn_operator.hpp>
