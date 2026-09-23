#pragma once

#include "../detail/ownership.h"

#include <atlas/geometry/sphere.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_sphere(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "Sphere", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("Sphere") = type;

    type.def(nb::new_([](const Float3& center, const float radius) {
            return PyGeometry {
                Geometry(Sphere::builder().with_center(center).with_radius(radius).build()),
                {}
            };
        }),
        "center"_a,
        "radius"_a,
        "A sphere shape wrapped as a Geometry.");
}

}
