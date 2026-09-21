#pragma once

#include "../../_detail/bvh.h"

namespace atlas::python {

inline void
register_bvh(nb::module_& m) {
    nb::class_<BVH>(m, "BVH")
        .def("build", &build_bvh, "triangles"_a);
}

}
