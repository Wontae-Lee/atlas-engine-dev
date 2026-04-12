#pragma once

/**
 * @file geometry.h
 * @brief Declares the abstract geometry interface used throughout Atlas.
 *
 * Geometry objects expose closest-point queries, signed-distance evaluation,
 * containment tests, bounding volumes, and a value-type GeometryOperator for
 * backend-portable runtime use.
 */

#include <atlas/geometry/geometry_type.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <type_traits>

namespace atlas::geometry {

template <typename T>
struct GeometryOperator;

/**
 * @brief Abstract base class for queryable geometric primitives.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Geometry {
    static_assert(std::is_floating_point_v<T>, "Geometry requires a floating-point T");

public:
    Geometry() = default;

    virtual ~Geometry() = default;

    /**
     * @brief Creates a value-type operator for host/device runtime queries.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE virtual atlas::geometry::GeometryOperator<T>
    make_geometry_operator() const = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    closest_point(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    closest_normal(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual T
    signed_distance(const atlas::math::Vector<T, 3>& p) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_inside(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_on_surface(const atlas::math::Vector<T, 3>& p, T tolerance) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::math::Vector<T, 3>
    centroid() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::spatial::AxisAlignedBoundingBox<T>
    bound() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual GeometryType
    type() const noexcept = 0;
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::geometry::GeometryOperator.
 */
template <typename T>
using GeometryOperator = geometry::GeometryOperator<T>;

/**
 * @brief Convenience alias for atlas::geometry::Geometry.
 */
template <typename T>
using Geometry = geometry::Geometry<T>;

template <typename T>
using GeometryHostPtr = atlas::host_shared_ptr<geometry::Geometry<T>>;

template <typename T>
using GeometryDevicePtr = atlas::device_shared_ptr<geometry::Geometry<T>>;

}

namespace atlas::spatial {

}
