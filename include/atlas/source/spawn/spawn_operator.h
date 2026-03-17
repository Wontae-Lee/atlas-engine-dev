// atlas/source/spawn/spawn_operator.h
#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/logging/logging.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/uniform_real_distribution.h>
#include <atlas/sampling/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace atlas::system {

enum class SpawnType : int {
    Surface,
    Volume
};

template <typename T>
struct BoxSpawnOperator final {
    const Vector3<T>* lower_corner = nullptr;
    const Vector3<T>* upper_corner = nullptr;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count, SpawnType type, std::uint32_t seed) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count, SpawnType type, std::uint32_t seed) const;

private:
    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static T
    uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1));

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_volume(const atlas::geometry::Box<T>& box, Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_surface(const atlas::geometry::Box<T>& box, Engine& engine);
};

template <typename T>
struct SphereSpawnOperator final {
    const Vector3<T>* center = nullptr;
    const T* radius          = nullptr;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count, SpawnType type, std::uint32_t seed) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count, SpawnType type, std::uint32_t seed) const;

private:
    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static T
    uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1));

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_unit_direction(Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_surface(const atlas::geometry::Sphere<T>& sphere, Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_volume(const atlas::geometry::Sphere<T>& sphere, Engine& engine);
};

template <typename T>
struct CylinderSpawnOperator final {
    const Vector3<T>* center = nullptr;
    const T* radius          = nullptr;
    const T* height          = nullptr;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count, SpawnType type, std::uint32_t seed) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count, SpawnType type, std::uint32_t seed) const;

private:
    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static T
    uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1));

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_disk(const Vector3<T>& center,
                const Vector3<T>& tangent,
                const Vector3<T>& bitangent,
                T radius,
                Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_volume(const atlas::geometry::Cylinder<T>& cylinder, Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_surface(const atlas::geometry::Cylinder<T>& cylinder, Engine& engine);
};

template <typename T>
struct TriangleSpawnOperator final {
    const Vector3<T>* a = nullptr;
    const Vector3<T>* b = nullptr;
    const Vector3<T>* c = nullptr;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count, SpawnType type, std::uint32_t seed) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count, SpawnType type, std::uint32_t seed) const;

private:
    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static T
    uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1));

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_surface(const atlas::geometry::Triangle<T>& triangle, Engine& engine);
};

template <typename T>
struct TriangleMeshSpawnOperator final {
    const atlas::geometry::TriangleMesh<T>* mesh       = nullptr;
    const HostBuffer<TriangleContainer4<T>>* triangles = nullptr;
    std::size_t max_volume_rejection_iterations        = 64;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count, SpawnType type, std::uint32_t seed) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count, SpawnType type, std::uint32_t seed) const;

private:
    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static T
    uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1));

    ATLAS_HOST ATLAS_FORCE_INLINE static T
    triangle_area(const TriangleContainer4<T>& tri);

    ATLAS_HOST ATLAS_FORCE_INLINE static bool
    is_finite_vector(const Vector3<T>& v) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static bool
    has_finite_bounds(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds) noexcept;

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_aabb(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds, Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_surface(const atlas::geometry::TriangleMesh<T>& mesh,
                   const HostBuffer<T>& prefix,
                   T total_area,
                   Engine& engine);

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    sample_volume(std::size_t count, Engine& engine) const;
};

template <typename T>
struct PlaneSpawnOperator final {
    const Vector3<T>* normal = nullptr;
    const T* offset          = nullptr;
    T plane_surface_extent   = T(1);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count, SpawnType type, std::uint32_t seed) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count, SpawnType type, std::uint32_t seed) const;

private:
    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static T
    uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1));

    template <typename Engine>
    ATLAS_HOST ATLAS_FORCE_INLINE static Vector3<T>
    sample_surface(const atlas::geometry::Plane<T>& plane, T patch_extent, Engine& engine);
};

template <typename T>
struct SpawnOperator final {
    static_assert(std::is_floating_point_v<T>, "SpawnOperator requires a floating-point T");

    SpawnType type = SpawnType::Surface;
    std::uint32_t seed = 5489u;
    T plane_surface_extent = T(1);
    std::size_t max_volume_rejection_iterations = 64;
    atlas::geometry::GeometryType geometry_type = atlas::geometry::GeometryType::Sphere;

    union {
        SphereSpawnOperator<T> sphere;
        CylinderSpawnOperator<T> cylinder;
        PlaneSpawnOperator<T> plane;
        BoxSpawnOperator<T> box;
        TriangleSpawnOperator<T> triangle;
        TriangleMeshSpawnOperator<T> triangle_mesh;
    };

    ATLAS_HOST ATLAS_FORCE_INLINE
    SpawnOperator() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SpawnOperator(SpawnType type_,
                  std::uint32_t seed_ = 5489u,
                  T plane_surface_extent_ = T(1)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE
    SpawnOperator(const SpawnOperator& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SpawnOperator&
    operator=(const SpawnOperator& other) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE ~SpawnOperator() noexcept;

    ATLAS_HOST
    SpawnOperator(const SphereSpawnOperator<T>& op,
                  SpawnType type_ = SpawnType::Surface,
                  std::uint32_t seed_ = 5489u) noexcept;

    ATLAS_HOST
    SpawnOperator(const CylinderSpawnOperator<T>& op,
                  SpawnType type_ = SpawnType::Surface,
                  std::uint32_t seed_ = 5489u) noexcept;

    ATLAS_HOST
    SpawnOperator(const PlaneSpawnOperator<T>& op,
                  SpawnType type_ = SpawnType::Surface,
                  std::uint32_t seed_ = 5489u) noexcept;

    ATLAS_HOST
    SpawnOperator(const BoxSpawnOperator<T>& op,
                  SpawnType type_ = SpawnType::Surface,
                  std::uint32_t seed_ = 5489u) noexcept;

    ATLAS_HOST
    SpawnOperator(const TriangleSpawnOperator<T>& op,
                  SpawnType type_ = SpawnType::Surface,
                  std::uint32_t seed_ = 5489u) noexcept;

    ATLAS_HOST
    SpawnOperator(const TriangleMeshSpawnOperator<T>& op,
                  SpawnType type_ = SpawnType::Surface,
                  std::uint32_t seed_ = 5489u) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(std::size_t count) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(const atlas::geometry::Geometry<T>& geometry, std::size_t count) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    spawn(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry, std::size_t count) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(std::size_t count) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(const atlas::geometry::Geometry<T>& geometry, std::size_t count) const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
    operator()(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry, std::size_t count) const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    copy_from(const SpawnOperator& other) noexcept;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using SpawnOperator = system::SpawnOperator<T>;

using SpawnType = system::SpawnType;

} // namespace atlas

#include <atlas/source/spawn/spawn_operator.hpp>