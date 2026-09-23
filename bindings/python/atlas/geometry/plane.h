#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/plane.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_plane(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "Plane", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("Plane") = type;

    type.def(nb::new_([](const Float3& normal, const float offset) {
            return PyGeometry {
                Geometry(Plane::builder().with_normal_offset(normal, offset).build()),
                {}
            };
        }),
        "normal"_a,
        "offset"_a,
        "An infinite plane with the given normal and signed offset.");

    type.def(nb::new_([](const Float3& point, const Float3& normal) {
            return PyGeometry {
                Geometry(Plane::builder().with_point_normal(point, normal).build()),
                {}
            };
        }),
        "point"_a,
        "normal"_a,
        "An infinite plane through a point with the given normal.");
}

}
