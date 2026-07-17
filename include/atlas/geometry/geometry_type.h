#pragma once

namespace atlas {

/**
 * @brief Discriminant tag identifying which leaf a `Geometry` tagged union holds.
 *
 * Each enumerator names one concrete geometry leaf and is paired, in
 * `geometry.h`, with the matching union member through a `DeviceVariantCase`.
 * The underlying type is fixed to `int` so the tag has a stable size and can be
 * copied to the device inside a trivially-copyable `Geometry`. `sphere` is the
 * default/fallback tag: a `Geometry` default-constructs to it, and any tag that
 * is not one of the registered cases normalizes to it.
 *
 * @note The enumerators are not assigned explicit values; their integer values
 *       follow declaration order and carry no external meaning.
 *       Dispatch is by tag equality, never by arithmetic on the value.
 */
enum class GeometryType : int {

    box, ///< Axis-defined rectangular solid leaf (`Box`).

    circle, ///< Flat filled disk leaf (`Circle`).

    cylinder, ///< Finite cylindrical solid leaf (`Cylinder`).

    plane, ///< Infinite half-space plane leaf (`Plane`).

    sphere, ///< Ball leaf (`Sphere`); the default and fallback case.

    square, ///< Flat filled quadrilateral leaf (`Square`).

    triangle, ///< Single flat triangle leaf (`Triangle`).

    triangle_mesh, ///< Triangle-soup mesh leaf, stored as a `TriangleMeshView`.

    polygonal_prism ///< Closed regular polygonal prism leaf (`PolygonalPrism`).
};

}
