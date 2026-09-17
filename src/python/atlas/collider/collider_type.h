#pragma once

#include "../_detail/boundary.h"

namespace atlas::python {

inline void
register_collider_type(nb::module_& m) {
    nb::enum_<ColliderType>(m, "ColliderType")
        .value("isothermal", ColliderType::isothermal);
}

}
