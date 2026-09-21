#pragma once

#include "../_detail/boundary.h"

namespace atlas::python {

inline void
register_isothermal_collider(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PyCollider>>(
        nb::module_::import_("builtins").attr("type")(
            "IsothermalCollider", nb::make_tuple(m.attr("Collider")), attributes));
    m.attr("IsothermalCollider") = type;

    type.def(nb::new_([](PyUnit unit,
           const float momentum_accommodation_coefficient,
           const float restitution,
           const DiffuseSampling diffuse_sampling) {
            Collider collider(IsothermalCollider::builder()
                                  .with_unit(std::move(unit.value))
                                  .with_momentum_accommodation_coefficient(
                                      momentum_accommodation_coefficient)
                                  .with_restitution(restitution)
                                  .with_diffuse_sampling(diffuse_sampling)
                                  .build());
            return PyCollider { std::move(collider), std::move(unit.mesh_owners) };
        }),
        "unit"_a,
        "momentum_accommodation_coefficient"_a = 1.0f,
        "restitution"_a                        = 1.0f,
        "diffuse_sampling"_a                   = DiffuseSampling::uniform,
        "An isothermal moving-wall collider wrapped as a Collider.");
}

}
