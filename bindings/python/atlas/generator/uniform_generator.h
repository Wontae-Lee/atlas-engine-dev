#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/generator/generator.h>
#include <atlas/generator/uniform_generator.h>
#include <atlas/math/vector/float3.h>
#include <atlas/memory/memory.h>
#include <atlas/random/seed.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_uniform_generator(nb::module_& m) {
    nb::dict attributes;
    attributes["__module__"] = m.attr("__name__");
    auto type = nb::borrow<nb::class_<Generator>>(
        nb::module_::import_("builtins").attr("type")(
            "UniformGenerator", nb::make_tuple(m.attr("Generator")), attributes));
    m.attr("UniformGenerator") = type;

    type.def(nb::new_([](std::vector<float> species_ratios,
           std::vector<float>
               species_numbers,
           const float temperature,
           const float min_value,
           const float max_value,
           const Float3& bulk_velocity,
           const unsigned int seed) {
            return atlas::make_host_shared<atlas::Generator>(atlas::Generator(
                atlas::UniformGenerator::builder()
                    .with_species_ratios(atlas::HostBuffer<float>(species_ratios.begin(), species_ratios.end()))
                    .with_species_numbers(atlas::HostBuffer<float>(species_numbers.begin(), species_numbers.end()))
                    .with_temperature(temperature)
                    .with_min_value(min_value)
                    .with_max_value(max_value)
                    .with_bulk_velocity(bulk_velocity)
                    .with_seed(seed)
                    .build()));
        }),
        "species_ratios"_a,
        "species_numbers"_a,
        "temperature"_a   = 273.15f,
        "min_value"_a     = 0.0f,
        "max_value"_a     = 0.0f,
        "bulk_velocity"_a = Float3 {},
        "seed"_a          = atlas::DEFAULT_UNSIGNED_INT_SEED,
        "Samples each velocity component uniformly over [min_value, max_value] "
        "plus a bulk drift; species chosen by weighted draw (temperature unused).");
}

}
