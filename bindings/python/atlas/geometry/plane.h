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
