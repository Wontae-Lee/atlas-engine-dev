#pragma once

#include "../_detail/boundary.h"

namespace atlas::python {

inline void
register_tracing_sink(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<PySink>>(
        nb::module_::import_("builtins").attr("type")(
            "TracingSink", nb::make_tuple(m.attr("Sink")), attributes));
    m.attr("TracingSink") = type;

    type.def(nb::new_([](PyUnit unit) {
            Sink sink(TracingSink::builder().with_unit(std::move(unit.value)).build());
            return PySink { std::move(sink), std::move(unit.mesh_owners) };
        }),
        "unit"_a,
        "A swept sink that despawns particles crossing the unit this step, wrapped as a Sink.");
}

}
