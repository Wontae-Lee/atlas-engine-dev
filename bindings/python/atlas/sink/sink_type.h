#pragma once

#include "sink.h"

namespace atlas::python {

inline void
register_sink_type(nb::module_& m) {
    nb::enum_<SinkType>(m, "SinkType")
        .value("surface", SinkType::surface)
        .value("volume", SinkType::volume)
        .value("tracing", SinkType::tracing);
}

}
