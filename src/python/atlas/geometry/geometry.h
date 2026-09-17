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
register_geometry(nb::module_& m) {
    nb::class_<PyGeometry>(m, "Geometry")
        .def(nb::init<>())
        .def(nb::init<const PyGeometry&>(), "other"_a)
        .def_prop_ro("type", [](const PyGeometry& geometry) { return geometry.value.type; })
        .def(
            "closest_point",
            [](const PyGeometry& geometry, const Float3& point) {
                return geometry.value.closest_point(point);
            },
            "point"_a)
        .def(
            "closest_normal",
            [](const PyGeometry& geometry, const Float3& point) {
                return geometry.value.closest_normal(point);
            },
            "point"_a)
        .def(
            "signed_distance",
            [](const PyGeometry& geometry, const Float3& point) {
                return geometry.value.signed_distance(point);
            },
            "point"_a)
        .def(
            "is_inside",
            [](const PyGeometry& geometry, const Float3& point, float tolerance) {
                return geometry.value.is_inside(point, tolerance);
            },
            "point"_a,
            "tolerance"_a = 0.0f)
        .def(
            "is_on_surface",
            [](const PyGeometry& geometry, const Float3& point, float tolerance) {
                return geometry.value.is_on_surface(point, tolerance);
            },
            "point"_a,
            "tolerance"_a = 0.0f)
        .def("centroid", [](const PyGeometry& geometry) { return geometry.value.centroid(); })
        .def("bound", [](const PyGeometry& geometry) { return geometry.value.bound(); })
        .def("is_valid", [](const PyGeometry& geometry) { return geometry.value.is_valid(); })
        .def(
            "trace",
            [](const PyGeometry& geometry, const Ray& ray) {
                return geometry.value.trace(ray);
            },
            "ray"_a)
        .def(
            "winding_number",
            [](const PyGeometry& geometry, const Float3& point) {
                if (geometry.value.type != GeometryType::triangle_mesh)
                    throw nb::type_error("winding_number requires a triangle mesh");
                return geometry.value.triangle_mesh.winding_number(point);
            },
            "point"_a);
}

}
