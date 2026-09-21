#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/polygonal_prism.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/math/vector/float3.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_geometry_type(nb::module_& m) {
    nb::enum_<GeometryType>(m, "GeometryType")
        .value("box", GeometryType::box)
        .value("circle", GeometryType::circle)
        .value("cylinder", GeometryType::cylinder)
        .value("plane", GeometryType::plane)
        .value("sphere", GeometryType::sphere)
        .value("square", GeometryType::square)
        .value("triangle", GeometryType::triangle)
        .value("triangle_mesh", GeometryType::triangle_mesh)
        .value("polygonal_prism", GeometryType::polygonal_prism);
}

}
