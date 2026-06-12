#pragma once

/**
 * @file geometry_type.h
 * @brief Declares the enumeration of all supported geometry primitive types.
 *
 * @details
 * This header defines @ref atlas::GeometryType, an enumeration used
 * to identify the concrete type of a geometry primitive at runtime.
 *
 * ## Purpose
 * The geometry type tag is primarily used in:
 * - @ref GeometryOperator as a discriminator for tagged-union dispatch,
 * - polymorphic geometry systems where dynamic type inspection is required,
 * - serialization, debugging, or logging of geometry instances,
 * - switch-based dispatch in performance-critical code paths.
 *
 * ## Design considerations
 * - The enumeration values map directly to supported geometry primitives in Atlas.
 * - Each value corresponds to a concrete geometry implementation and its
 *   associated operator type.
 * - The order and values are stable identifiers and may be relied upon by
 *   internal systems (e.g., switch dispatch or compact encoding).
 *
 * ## Extensibility
 * When introducing a new geometry primitive:
 * - a corresponding entry must be added to this enumeration,
 * - the @ref GeometryOperator union must be extended accordingly,
 * - dispatch logic must be updated to handle the new type.
 *
 * ---
 */

namespace atlas {

/**
 * @brief Identifies the concrete geometry primitive type at runtime.
 *
 * @details
 * This enumeration is used as a type tag for:
 * - selecting the active variant in @ref GeometryOperator,
 * - enabling efficient runtime dispatch without virtual functions,
 * - distinguishing between different geometry implementations in a uniform API.
 *
 * Each enumerator corresponds to a specific primitive family supported by Atlas.
 */
enum class GeometryType : int {

    /**
     * @brief Axis-aligned box geometry (AABB).
     *
     * Represents a rectangular prism aligned with the coordinate axes.
     */
    Box,

    /**
     * @brief Circular disk geometry.
     *
     * Represents a circle embedded in 3D space, typically defined by a center,
     * normal, and radius.
     */
    Circle,

    /**
     * @brief Finite cylinder geometry.
     *
     * Represents a cylinder defined by a center position, radius, and height.
     */
    Cylinder,

    /**
     * @brief Infinite plane geometry.
     *
     * Represents a plane defined by a point and a normal direction.
     */
    Plane,

    /**
     * @brief Sphere geometry.
     *
     * Represents a solid sphere defined by a center and radius.
     */
    Sphere,

    /**
     * @brief Square surface geometry.
     *
     * Represents an oriented finite square defined by a center, normal,
     * and side length.
     */
    Square,

    /**
     * @brief Triangle geometry.
     *
     * Represents a single triangular surface primitive defined by three vertices.
     */
    Triangle,

    /**
     * @brief Triangle mesh geometry.
     *
     * Represents a collection of triangles forming a surface or volume boundary.
     */
    TriangleMesh
};

} // namespace atlas
