#pragma once

#include "collider.h"

#include <atlas/collider/diffuse_sampling.h>

namespace atlas::python {

inline void
register_diffuse_sampling(nb::module_& m) {
    nb::enum_<DiffuseSampling>(m, "DiffuseSampling")
        .value("cosine_weighted", DiffuseSampling::cosine_weighted)
        .value("uniform", DiffuseSampling::uniform);
}

}
