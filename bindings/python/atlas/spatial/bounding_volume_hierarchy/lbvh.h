#pragma once

#include "bvh.h"

#include <atlas/spatial/bounding_volume_hierarchy/lbvh.h>

namespace atlas::python {

inline void
register_lbvh(nb::module_& m) {
    nb::class_<LBVH, BVH>(m, "LBVH")
        .def(nb::new_([](const std::vector<PyGeometry>& triangles, int morton_bits) {
                 auto hierarchy = atlas::make_host_shared<LBVH>();
                 hierarchy->set_morton_bits(morton_bits);
                 build_bvh(*hierarchy, triangles);
                 return hierarchy;
             }),
             "triangles"_a   = std::vector<PyGeometry>(),
             "morton_bits"_a = 10)
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
}

}
