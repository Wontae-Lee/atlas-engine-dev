#pragma once

/**
 * @file transformed_bounds.h
 * @brief Declares helpers for transforming spatial bounds.
 */

#include <atlas/spatial/axis_aligned_bounding_box.h>

#include <cstddef>

namespace atlas {

/**
 * @brief Transforms an AABB by applying a point transform to all eight corners.
 *
 * @tparam T Floating-point scalar type.
 * @tparam TransformPoint Callable returning a transformed Vector3<T>.
 *
 * @param bound Input axis-aligned bound in the source coordinate space.
 * @param transform Point transform applied to each source-space corner.
 * @return World-space AABB enclosing all transformed corners, or an invalid box
 *         when @p bound is invalid.
 */
template <typename T, typename TransformPoint>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AxisAlignedBoundingBox<T>
transform_aabb(const AxisAlignedBoundingBox<T>& bound,
               TransformPoint transform) noexcept;

} // namespace atlas

namespace atlas {

using atlas::transform_aabb;

} // namespace atlas

#include <atlas/spatial/transformed_bounds.hpp>
