#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/cylinder.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_cylinder(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "Cylinder", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("Cylinder") = type;

    type.def(nb::new_([](const Float3& center, const float radius, const float height, const bool open) {
            return PyGeometry {
                Geometry(Cylinder::builder()
                             .with_center(center)
                             .with_radius(radius)
                             .with_height(height)
                             .with_open(open)
                             .build()),
                {}
            };
        }),
        "center"_a,
        "radius"_a,
        "height"_a,
        "open"_a = false,
        "A cylinder; open=True drops the end caps.");
}

}
