#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/polygonal_prism.h>

#include <nanobind/nanobind.h>

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
