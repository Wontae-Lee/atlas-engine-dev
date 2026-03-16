#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/logging/logging.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Sampling domain used by @ref SpawnOperator.
 *
 * @details
 * - `Surface` samples points on the geometry boundary/surface.
 * - `Volume` samples points inside the geometry volume when the geometry has
 *   a meaningful interior.
 */
enum class SpawnType : int {
    Surface,
    Volume
};

/**
 * @brief Host-side geometry sampler that emits spawn positions.
 *
 * @details
 * `SpawnOperator` converts a geometry object into a host buffer of spawn
 * positions. Dispatch is driven by `geometry.type()` and the operator's
 * configured @ref SpawnType.
 *
 * Supported behavior:
 * - `Box`, `Sphere`, `Cylinder`:
 *   - surface and volume sampling
 * - `Triangle`, `TriangleMesh`:
 *   - surface sampling
 *   - volume sampling for `TriangleMesh` is best-effort rejection sampling
 *     against the mesh signed-distance field and assumes the mesh behaves like
 *     a closed surface
 * - `Plane`:
 *   - surface sampling over a finite square patch centered on the plane
 *   - volume sampling is unsupported and returns an empty buffer
 *
 * The operator is intentionally host-only because it constructs a
 * @ref HostBuffer and uses host random-number generators.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SpawnOperator final {
    static_assert(std::is_floating_point_v<T>, "SpawnOperator requires a floating-point T");

    /// @brief Which geometric domain to sample from.
    SpawnType type = SpawnType::Surface;

    /// @brief Seed forwarded to the backend-selected host random engine.
    std::uint32_t seed = 5489u;

    /**
     * @brief Side length half-extent used when sampling an infinite plane.
     *
     * @details
     * Plane sampling cannot cover an infinite domain, so the operator samples a
     * finite square patch centered on a representative point on the plane:
     * `point_on_plane +/- plane_surface_extent * tangent/bitangent`.
     */
    T plane_surface_extent = T(1);

    /**
     * @brief Multiplier controlling rejection attempts for volume fallback paths.
     *
     * @details
     * Used primarily for triangle-mesh volume sampling. The operator attempts at
     * most `count * max_volume_rejection_iterations` random points inside the
     * geometry bounding box before returning whatever was accepted.
     */
    std::size_t max_volume_rejection_iterations = 64;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SpawnOperator() noexcept = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SpawnOperator(SpawnType type_,
                  std::uint32_t seed_ = 5489u,
                  T plane_surface_extent_ = T(1)) noexcept;

    /**
     * @brief Generate spawn positions for a concrete geometry object.
     *
     * @param geometry Geometry to sample.
     * @param count Number of points requested.
     * @return Host buffer containing the generated positions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(const atlas::geometry::Geometry<T>& geometry, std::size_t count) const;

    /**
     * @brief Generate spawn positions from a shared geometry pointer.
     *
     * @param geometry Shared host-side geometry handle.
     * @param count Number of points requested.
     * @return Empty buffer if `geometry == nullptr`, otherwise the generated positions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry, std::size_t count) const;

    /**
     * @brief Convenience call operator forwarding to @ref spawn.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(const atlas::geometry::Geometry<T>& geometry, std::size_t count) const;

    /**
     * @brief Convenience call operator forwarding to @ref spawn.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry, std::size_t count) const;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using SpawnOperator = system::SpawnOperator<T>;

using SpawnType = system::SpawnType;

} // namespace atlas

#include <atlas/source/spawn/spawn_operator.hpp>
