#pragma once

#include <atlas/codec/codec.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_codec_type(nb::module_& m) {
    nb::enum_<CodecType>(m, "CodecType")
        .value("knudsen", CodecType::knudsen);
}

}
