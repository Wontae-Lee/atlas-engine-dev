#pragma once

/**
 * @file geometry_type.h
 * @brief The tag enum selecting which concrete shape a
 *        `Geometry`/`Geometry` value holds; see
 *        `geometry.h` for the tagged-union dispatch pattern
 *        this drives.
 */

namespace atlas {

/** @brief Which concrete shape a `Geometry`/`Geometry` is. */
enum class GeometryType : int {

    box,

    circle,

    cylinder,

    plane,

    sphere,

    square,

    triangle,

    /** Arbitrary triangle mesh; see `triangle_mesh.h`. */
    triangle_mesh
};

}