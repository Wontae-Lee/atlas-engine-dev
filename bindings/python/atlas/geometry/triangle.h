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
register_triangle(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyGeometry>>(
        nb::module_::import_("builtins").attr("type")(
            "Triangle", nb::make_tuple(m.attr("Geometry")), attributes));
    m.attr("Triangle") = type;

    type.def(nb::new_([](const Float3& a,
           const Float3& b,
           const Float3& c,
           const std::optional<Float3>& normal) {
            auto builder = Triangle::builder().with_vertices(a, b, c);
            if (normal.has_value()) {
                builder.with_normal(*normal);
            }
            return PyGeometry { Geometry(builder.build()), {} };
        }),
        "a"_a,
        "b"_a,
        "c"_a,
        "normal"_a = nb::none(),
        "A triangle with a derived normal unless an explicit normal is supplied.");
}

}
