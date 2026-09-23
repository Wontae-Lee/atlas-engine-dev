#pragma once

#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/sync/sync.h>

#include <nanobind/nanobind.h>

#include <cstddef>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_axis_aligned_bounding_box(nb::module_& m) {
    nb::class_<AABB>(m, "AABB")
        .def(nb::init<>())
        .def(nb::init<const AABB&>(), "other"_a)
        .def(nb::init<const Float3&, const Float3&>(), "lower"_a, "upper"_a)
        .def_rw("lower_corner", &AABB::lower_corner)
        .def_rw("upper_corner", &AABB::upper_corner)
        .def("area", &AABB::area)
        .def("width", &AABB::width)
        .def("height", &AABB::height)
        .def("depth", &AABB::depth)
        .def(
            "length",
            [](const AABB& bounds, int axis) {
                if (axis < 0 || axis >= 3) throw nb::index_error("AABB axis out of range");
                return bounds.length(static_cast<std::size_t>(axis));
            },
            "axis"_a)
        .def("overlaps", &AABB::overlaps, "other"_a)
        .def("contains", &AABB::contains, "point"_a)
        .def("intersects", &AABB::intersects, "ray"_a)
        .def("trace", &AABB::trace, "ray"_a)
        .def("center", &AABB::center)
        .def("extents", &AABB::extents)
        .def("diagonal_length", &AABB::diagonal_length)
        .def("diagonal_length_squared", &AABB::diagonal_length_squared)
        .def("is_valid", &AABB::is_valid)
        .def("is_empty", &AABB::is_empty)
        .def("reset", &AABB::reset)
        .def(
            "merge",
            [](AABB& bounds, const Float3& point) { bounds.merge(point); },
            "point"_a)
        .def(
            "merge",
            [](AABB& bounds, const AABB& other) { bounds.merge(other); },
            "other"_a)
        .def("expand", &AABB::expand, "delta"_a)
        .def(
            "corner",
            [](const AABB& bounds, int index) {
                if (index < 0 || index >= 8) throw nb::index_error("AABB corner out of range");
                return bounds.corner(static_cast<std::size_t>(index));
            },
            "index"_a)
        .def("clamp", &AABB::clamp, "point"_a);

    m.def("make_aabb", &atlas::make_aabb, "point"_a);

    m.def("merge_aabb", &atlas::merge_aabb, "a"_a, "b"_a);

    m.def("aabb_distance_squared", &atlas::aabb_distance_squared, "bounds"_a, "point"_a);

    m.def(
        "transform_aabb",
        [](const AABB& bound, const Sync& pose) {
            return atlas::transform_aabb(bound, [&pose](const Float3& point) {
                return pose.sync_to_world(point);
            });
        },
        "bound"_a,
        "sync"_a);

    m.def(
        "transform_aabb",
        [](const AABB& bound, const nb::callable& transform) {
            AABB transformed;
            if (!bound.is_valid()) return transformed;
            for (std::size_t corner = 0; corner < 8; ++corner)
                transformed.merge(nb::cast<Float3>(transform(bound.corner(corner))));
            return transformed;
        },
        "bound"_a,
        "transform"_a);
}

}
