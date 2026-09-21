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
register_ray(nb::module_& m) {
    nb::class_<Ray>(m, "Ray")
        .def(nb::init<>())
        .def(nb::init<const Ray&>(), "other"_a)
        .def(nb::init<const Float3&, const Float3&>(), "origin"_a, "direction"_a)
        .def_rw("origin", &Ray::origin)
        .def_rw("direction", &Ray::direction)
        .def("point_at", &Ray::point_at, "distance"_a);

    nb::class_<HitSurface>(m, "HitSurface")
        .def(nb::init<>())
        .def_rw("is_intersecting", &HitSurface::is_intersecting)
        .def_rw("distance", &HitSurface::distance)
        .def_rw("point", &HitSurface::point)
        .def_rw("normal", &HitSurface::normal);

    nb::class_<HitAABB>(m, "HitAABB")
        .def(nb::init<>())
        .def_rw("is_intersecting", &HitAABB::is_intersecting)
        .def_rw("enter", &HitAABB::enter)
        .def_rw("exit", &HitAABB::exit);

    m.def(
        "ray_plane_distance",
        [](const Float3& point, const Float3& normal, const Ray& ray)
            -> std::optional<float> {
            float distance;
            if (!atlas::ray_plane_distance(point, normal, ray, distance)) return std::nullopt;
            return distance;
        },
        "plane_point"_a,
        "plane_normal"_a,
        "ray"_a);
}

}
