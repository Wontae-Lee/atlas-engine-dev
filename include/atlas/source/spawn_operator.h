#pragma once

#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Spawn classification mode used by @ref SpawnOperator.
 *
 * @tparam T Floating-point scalar type.
 */
enum class SpawnType : int {
    Surface,
    Volume
};

/**
 * @brief Surface spawn predicate for a single sample position.
 *
 * @details
 * Returns `true` when the sample should be accepted by surface classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceSpawnOperator final {
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) const noexcept;
};

/**
 * @brief Volume spawn predicate for a single sample position.
 *
 * @details
 * Returns `true` when the sample should be accepted by volume classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct VolumeSpawnOperator final {
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    spawn(const atlas::geometry::GeometryOperator<T>& query,
          const Vector3<T>& particle,
          T tolerance = T(0)) const noexcept;
};

/**
 * @brief Runtime-dispatched spawn predicate for surface or volume classification.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SpawnOperator final {
    SpawnType type = SpawnType::Surface;
    union {
        SurfaceSpawnOperator<T> surface;
        VolumeSpawnOperator<T> volume;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SpawnOperator() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    explicit SpawnOperator(SpawnType type) noexcept;

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

} // namespace atlas::system

#include <atlas/source/spawn_operator.hpp>
