#pragma once

#include "../_detail/boundary.h"

namespace atlas::python {

inline void
register_sink(nb::module_& m) {
    nb::class_<PySink>(m, "Sink")
        .def_prop_ro("type", [](const PySink& sink) { return sink.value.type; })
        .def_prop_ro("unit", [](const PySink& sink) {
            const Unit unit = SinkVariant::visit(sink.value, BoundaryUnit {}, Unit {});
            return PyUnit { unit, sink.mesh_owners };
        })
        .def(
            "advance",
            [](PySink& sink, const float dt) { sink.value.advance(dt); },
            "dt"_a)
        .def(
            "despawn",
            [](const PySink& sink, const Float3& position, const Float3& velocity, const float dt) {
                return sink.value.despawn(position, velocity, dt);
            },
            "position"_a,
            "velocity"_a,
            "dt"_a);
}

}
