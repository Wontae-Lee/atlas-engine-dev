#pragma once

#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_node(nb::module_& m) {
    nb::class_<BVHNode>(m, "BVHNode")
        .def(nb::init<>())
        .def(nb::init<const BVHNode&>(), "other"_a)
        .def_rw("bounds", &BVHNode::bounds)
        .def_rw("solid_angle_moment", &BVHNode::solid_angle_moment)
        .def_rw("solid_angle_normal_area", &BVHNode::solid_angle_normal_area)
        .def_rw("solid_angle_area", &BVHNode::solid_angle_area)
        .def_rw("left", &BVHNode::left)
        .def_rw("right", &BVHNode::right)
        .def_rw("start", &BVHNode::start)
        .def_rw("count", &BVHNode::count)
        .def_rw("is_leaf", &BVHNode::is_leaf);
}

}
