#pragma once

#include "../_detail/handles.h"

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
register_polygonal_prism(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "PolygonalPrism", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("PolygonalPrism") = type;

    type.def(nb::new_([](const Float3& center,
           const int side_count,
           const float radius,
           const float height) {
            return PyGeometry {
                Geometry(PolygonalPrism::builder()
                             .with_center(center)
                             .with_side_count(side_count)
                             .with_radius(radius)
                             .with_height(height)
                             .build()),
                {}
            };
        }),
        "center"_a,
        "side_count"_a,
        "radius"_a,
        "height"_a,
        "A closed regular polygonal prism aligned with the z axis.");
}

}
