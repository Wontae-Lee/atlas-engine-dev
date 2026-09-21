#pragma once

#include <atlas/codec/codec.h>
#include <atlas/codec/knudsen_codec.h>
#include <atlas/memory/memory.h>
#include <atlas/random/default_random_engine.h>
#include <atlas/random/seed.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/solver.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_knudsen_codec(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<Codec>>(
        nb::module_::import_("builtins").attr("type")(
            "KnudsenCodec", nb::make_tuple(m.attr("Codec")), attributes));
    m.attr("KnudsenCodec") = type;

    type.def(nb::new_([](const float representative_characteristic_length,
           const float representative_collision_cross_sectional_area,
           const float representative_statistical_weight,
           const float representative_cell_volume) {
            atlas::CodecHostPtr codec = atlas::make_host_shared<atlas::Codec>(atlas::Codec(
                atlas::KnudsenCodec::builder()
                    .with_representative_characteristic_length(representative_characteristic_length)
                    .with_representative_collision_cross_sectional_area(
                        representative_collision_cross_sectional_area)
                    .with_representative_statistical_weight(representative_statistical_weight)
                    .with_representative_cell_volume(representative_cell_volume)
                    .build()));
            return codec;
        }),
        "representative_characteristic_length"_a          = 1.0f,
        "representative_collision_cross_sectional_area"_a = 1.0f,
        "representative_statistical_weight"_a             = 1.0f,
        "representative_cell_volume"_a                    = 1.0f,
        "A Knudsen-number per-cell solver codec wrapped as a shared Codec handle. "
        "Every representative scalar must be positive.");
}

}
