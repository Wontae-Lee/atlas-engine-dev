#pragma once

#include "../_detail/boundary.h"

namespace atlas::python {

inline void
register_volume_sink(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PySink>>(
        nb::module_::import_("builtins").attr("type")(
            "VolumeSink", nb::make_tuple(m.attr("Sink")), attributes));
    m.attr("VolumeSink") = type;

    type.def(nb::new_([](PyUnit unit, const float tolerance) {
            Sink sink(VolumeSink::builder()
                          .with_unit(std::move(unit.value))
                          .with_tolerance(tolerance)
                          .build());
            return PySink { std::move(sink), std::move(unit.mesh_owners) };
        }),
        "unit"_a,
        "tolerance"_a = 0.0f,
        "A sink that despawns particles inside the unit's volume, wrapped as a Sink.");
}

}
