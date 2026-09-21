#pragma once

#include "../detail/ownership.h"

#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/generator/jittering_generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/generator/maxwell_sigma_generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>
#include <atlas/source/source.h>
#include <atlas/source/surface_source.h>
#include <atlas/source/volume_source.h>
#include <atlas/unit/unit.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_maxwell_sigma_generator(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<Generator>>(
        nb::module_::import_("builtins").attr("type")(
            "MaxwellSigmaGenerator", nb::make_tuple(m.attr("Generator")), attributes));
    m.attr("MaxwellSigmaGenerator") = type;

    type.def(nb::new_([](std::vector<float> species_ratios,
           std::vector<float>
               species_numbers,
           const float temperature,
           const float sigma,
           const Float3& bulk_velocity,
           const unsigned int seed) {
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(
                atlas::MaxwellSigmaGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_temperature(temperature)
                    .with_sigma(sigma)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed)
                    .build()));
        }),
        "species_ratios"_a,
        "species_numbers"_a,
        "temperature"_a   = 273.15f,
        "sigma"_a         = 0.0f,
        "bulk_velocity"_a = Float3 {},
        "seed"_a          = atlas::DEFAULT_UNSIGNED_INT_SEED,
        "Samples an isotropic Gaussian velocity with caller-supplied sigma and bulk drift.");
}

}
