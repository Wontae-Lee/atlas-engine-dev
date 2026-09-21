#pragma once

#include "../detail/ownership.h"

#include <atlas/math/vector/float3.h>
#include <atlas/sink/sink.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

namespace {

    struct SinkUnit final {
        template <typename Boundary>
        ATLAS_ALL_DEVICE Unit
        operator()(const Boundary& boundary) const noexcept {
            return boundary.unit();
        }
    };

}

inline void
register_sink(nb::module_& m) {
    nb::class_<PySink>(m, "Sink")
        .def_prop_ro("type", [](const PySink& sink) { return sink.value.type; })
        .def_prop_ro("unit", [](const PySink& sink) {
            const Unit unit = SinkVariant::visit(sink.value, SinkUnit {}, Unit {});
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
