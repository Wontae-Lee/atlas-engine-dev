#include "binding_types.h"
#include "register.h"

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>
#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {
namespace {

void
build_bvh(BVH& hierarchy, const std::vector<PyGeometry>& geometries) {
    if (geometries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
        throw nb::value_error("BVH exceeds the node index range");
    HostBuffer<TriangleContainer4> triangles(geometries.size());
    for (std::size_t i = 0; i < geometries.size(); ++i) {
        if (geometries[i].value.type != GeometryType::triangle)
            throw nb::type_error("Every BVH primitive must be a triangle geometry");
        const Triangle& triangle = geometries[i].value.triangle;
        triangles[i].a() = triangle.a;
        triangles[i].b() = triangle.b;
        triangles[i].c() = triangle.c;
        triangles[i].d() = triangle.normal;
    }
    hierarchy.build(triangles);
}

}

void
register_bvh(nb::module_& m) {
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

    nb::class_<BVH>(m, "BVH")
        .def("build", &build_bvh, "triangles"_a);

    nb::class_<LBVH, BVH>(m, "LBVH")
        .def(nb::new_([](const std::vector<PyGeometry>& triangles, int morton_bits) {
            auto hierarchy = atlas::make_host_shared<LBVH>();
            hierarchy->set_morton_bits(morton_bits);
            build_bvh(*hierarchy, triangles);
            return hierarchy;
        }), "triangles"_a = std::vector<PyGeometry>(), "morton_bits"_a = 10)
        .def("reset", &LBVH::reset)
        .def("set_morton_bits", &LBVH::set_morton_bits, "morton_bits"_a)
        .def_prop_rw("morton_bits", &LBVH::morton_bits, &LBVH::set_morton_bits)
        .def_prop_ro("root", &LBVH::root)
        .def("nodes", [](const LBVH& hierarchy) {
            return std::vector<BVHNode>(hierarchy.nodes().begin(), hierarchy.nodes().end());
        })
        .def("indices", [](const LBVH& hierarchy) {
            return std::vector<int>(hierarchy.indices().begin(), hierarchy.indices().end());
        })
        .def("bounds", [](const LBVH& hierarchy) {
            return std::vector<AABB>(hierarchy.bounds().begin(), hierarchy.bounds().end());
        })
        .def("centroids", [](const LBVH& hierarchy) {
            return std::vector<Float3>(hierarchy.centroids().begin(), hierarchy.centroids().end());
        });

    nb::class_<SAHBVH, BVH>(m, "SAHBVH")
        .def(nb::new_([](const std::vector<PyGeometry>& triangles, int leaf_size, int bin_count) {
            auto hierarchy = atlas::make_host_shared<SAHBVH>();
            hierarchy->set_leaf_size(leaf_size);
            hierarchy->set_bin_count(bin_count);
            build_bvh(*hierarchy, triangles);
            return hierarchy;
        }), "triangles"_a = std::vector<PyGeometry>(), "leaf_size"_a = 32, "bin_count"_a = 100)
        .def("reset", &SAHBVH::reset)
        .def("set_leaf_size", &SAHBVH::set_leaf_size, "leaf_size"_a)
        .def("set_bin_count", &SAHBVH::set_bin_count, "bin_count"_a)
        .def_prop_rw("leaf_size", &SAHBVH::leaf_size, &SAHBVH::set_leaf_size)
        .def_prop_rw("bin_count", &SAHBVH::bin_count, &SAHBVH::set_bin_count)
        .def_prop_ro("root", &SAHBVH::root)
        .def("nodes", [](const SAHBVH& hierarchy) {
            return std::vector<BVHNode>(hierarchy.nodes().begin(), hierarchy.nodes().end());
        })
        .def("indices", [](const SAHBVH& hierarchy) {
            return std::vector<int>(hierarchy.indices().begin(), hierarchy.indices().end());
        })
        .def("bounds", [](const SAHBVH& hierarchy) {
            return std::vector<AABB>(hierarchy.bounds().begin(), hierarchy.bounds().end());
        })
        .def("centroids", [](const SAHBVH& hierarchy) {
            return std::vector<Float3>(hierarchy.centroids().begin(), hierarchy.centroids().end());
        });
}

}
