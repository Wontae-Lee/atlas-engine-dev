#pragma once

#include <atlas/math/math.h>
#include <atlas/spatial/ray.h>

#include <cstddef>
#include <limits>

namespace atlas {

template <typename T>
struct AxisAlignedBoundingBoxRayIntersection {

    bool is_intersecting = false;

    T enter = T(0);

    T exit = std::numeric_limits<T>::max();
};

template <typename T>
class AxisAlignedBoundingBox final {
public:
    Vector3<T> lower_corner;

    Vector3<T> upper_corner;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const Vector3<T>& point1, const Vector3<T>& point2) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    AxisAlignedBoundingBox(const AxisAlignedBoundingBox& other) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    area() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    width() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    height() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    depth() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    length(std::size_t axis) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    overlaps(const AxisAlignedBoundingBox& other) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    contains(const Vector3<T>& point) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray<T>& ray) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBoxRayIntersection<T>
    trace(const Ray<T>& ray) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    center() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    extents() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    diagonal_length_squared() const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    reset() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const Vector3<T>& point) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    merge(const AxisAlignedBoundingBox& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    expand(T delta) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    corner(std::size_t idx) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    clamp(const Vector3<T>& point) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_empty() const noexcept;
};

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBox<T>
make_aabb(const Vector3<T>& p) noexcept {
    AxisAlignedBoundingBox<T> b;
    b.lower_corner = p;
    b.upper_corner = p;
    return b;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBox<T>
merge_aabb(const AxisAlignedBoundingBox<T>& a,
           const AxisAlignedBoundingBox<T>& b) noexcept {
    AxisAlignedBoundingBox<T> out;
    out.lower_corner = cmin(a.lower_corner, b.lower_corner);
    out.upper_corner = cmax(a.upper_corner, b.upper_corner);
    return out;
}

template <typename T>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
aabb_distance_squared(const AxisAlignedBoundingBox<T>& bounds,
                      const Vector3<T>& point) noexcept {
    const Vector3<T> closest = bounds.clamp(point);
    return (closest - point).length_squared();
}

template <typename T, typename TransformPoint>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBox<T>
transform_aabb(const AxisAlignedBoundingBox<T>& bound,
               TransformPoint transform) noexcept;

}

namespace atlas {

template <typename T>
using HitAABB = atlas::AxisAlignedBoundingBoxRayIntersection<T>;

template <typename T>
using AABB = atlas::AxisAlignedBoundingBox<T>;

using AABBF = AxisAlignedBoundingBox<float>;

using AABBD = AxisAlignedBoundingBox<double>;

using AABBRayInteractionF = AxisAlignedBoundingBoxRayIntersection<float>;

using AABBRayInteractionD = AxisAlignedBoundingBoxRayIntersection<double>;

}

#include <atlas/spatial/axis_aligned_bounding_box.hpp>