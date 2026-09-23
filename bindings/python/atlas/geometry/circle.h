#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/circle.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_circle(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "Circle", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("Circle") = type;

    type.def(nb::new_([](const Float3& center, const Float3& normal, const float radius) {
            return PyGeometry {
                Geometry(Circle::builder()
                             .with_center(center)
                             .with_normal(normal)
                             .with_radius(radius)
                             .build()),
                {}
            };
        }),
        "center"_a,
        "normal"_a,
        "radius"_a,
        "A filled disk with the given center, normal, and radius.");
}

}
