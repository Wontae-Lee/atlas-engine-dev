#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/square.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_square(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "Square", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("Square") = type;

    type.def(nb::new_([](const Float3& center, const Float3& normal, const float side_length) {
            return PyGeometry {
                Geometry(Square::builder()
                             .with_center(center)
                             .with_normal(normal)
                             .with_side_length(side_length)
                             .build()),
                {}
            };
        }),
        "center"_a,
        "normal"_a,
        "side_length"_a,
        "A filled square with the given center, normal, and side length.");
}

}
