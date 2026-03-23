#pragma once

namespace atlas::geometry {

/* ====================================================================== */
/* Tags                                                                    */
/* ====================================================================== */

/**
 * @brief Runtime type tag for @ref GeometryOperator.
 *
 * @details
 * Identifies which concrete operator inside the union is currently active.
 *
 * @note
 * Keep this enum in sync with the union members in @ref GeometryOperator.
 */
enum class GeometryType : int {
    /// @brief Axis-aligned box operator.
    Box,
    /// @brief Finite circle operator.
    Circle,
    /// @brief Finite cylinder operator (axis-aligned in its local frame).
    Cylinder,
    /// @brief Infinite plane operator.
    Plane,
    /// @brief Sphere operator.
    Sphere,
    /// @brief Single triangle operator.
    Triangle,
    /// @brief Triangle mesh operator (vertex/index arrays).
    TriangleMesh
};

}
