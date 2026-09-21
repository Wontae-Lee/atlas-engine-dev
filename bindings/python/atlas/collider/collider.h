#pragma once

#include "../detail/ownership.h"

#include <atlas/collider/collider.h>
#include <atlas/math/vector/float3.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

namespace {

    struct ColliderUnit final {
        template <typename Boundary>
        ATLAS_ALL_DEVICE Unit
        operator()(const Boundary& boundary) const noexcept {
            return boundary.unit();
        }
    };

}

inline void
register_collider(nb::module_& m) {
    nb::class_<PyCollider>(m, "Collider")
        .def_prop_ro("type", [](const PyCollider& collider) { return collider.value.type; })
        .def_prop_ro("unit", [](const PyCollider& collider) {
            const Unit unit = ColliderVariant::visit(collider.value, ColliderUnit {}, Unit {});
            return PyUnit { unit, collider.mesh_owners };
        })
        .def("bound", [](const PyCollider& collider) { return collider.value.bound(); })
        .def(
            "advance",
            [](PyCollider& collider, const float dt) { collider.value.advance(dt); },
            "dt"_a)
        .def(
            "trace",
            [](const PyCollider& collider, const Float3& position, const Float3& velocity, const float dt) {
                return collider.value.trace(position, velocity, dt);
            },
            "position"_a,
            "velocity"_a,
            "dt"_a)
        .def(
            "collide",
            [](const PyCollider& collider, const HitSurface& hit, Float3 position, Float3 velocity, const float dt) {
                collider.value.collide(hit, position, velocity, dt);
                return nb::make_tuple(position, velocity);
            },
            "hit"_a,
            "position"_a,
            "velocity"_a,
            "dt"_a,
            "Returns the post-collision position and velocity without modifying the input values.");
}

}
