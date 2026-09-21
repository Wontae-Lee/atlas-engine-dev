#pragma once

#include "../../detail/ownership.h"

#include <atlas/spatial/bounding_volume_hierarchy/bvh.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
build_bvh(BVH& hierarchy, const std::vector<PyGeometry>& geometries) {
    if (geometries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
        throw nb::value_error("BVH exceeds the node index range");
    }
    HostBuffer<TriangleContainer4> triangles(geometries.size());
    for (std::size_t i = 0; i < geometries.size(); ++i) {
        if (geometries[i].value.type != GeometryType::triangle) {
            throw nb::type_error("Every BVH primitive must be a triangle geometry");
        }
        const Triangle& triangle = geometries[i].value.triangle;
        triangles[i].a()         = triangle.a;
        triangles[i].b()         = triangle.b;
        triangles[i].c()         = triangle.c;
        triangles[i].d()         = triangle.normal;
    }
    hierarchy.build(triangles);
}

inline void
register_bvh(nb::module_& m) {
    nb::class_<BVH>(m, "BVH")
        .def("build", &build_bvh, "triangles"_a);
}

}
