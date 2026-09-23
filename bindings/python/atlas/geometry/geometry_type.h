#pragma once

#include <atlas/geometry/geometry.h>

#include <nanobind/nanobind.h>

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
