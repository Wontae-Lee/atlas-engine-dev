#pragma once

#include "bvh.h"

#include <atlas/spatial/bounding_volume_hierarchy/sah_bvh.h>

namespace atlas::python {

inline void
register_sah_bvh(nb::module_& m) {
    nb::class_<SAHBVH, BVH>(m, "SAHBVH")
        .def(nb::new_([](const std::vector<PyGeometry>& triangles, int leaf_size, int bin_count) {
                 auto hierarchy = atlas::make_host_shared<SAHBVH>();
                 hierarchy->set_leaf_size(leaf_size);
                 hierarchy->set_bin_count(bin_count);
                 build_bvh(*hierarchy, triangles);
                 return hierarchy;
             }),
             "triangles"_a = std::vector<PyGeometry>(),
             "leaf_size"_a = 32,
             "bin_count"_a = 100)
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
