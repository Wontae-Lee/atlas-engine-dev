#pragma once

#include <atlas/generator/generator.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_generator_type(nb::module_& m) {
    nb::enum_<GeneratorType>(m, "GeneratorType")
        .value("uniform", GeneratorType::uniform)
        .value("jittering", GeneratorType::jittering)
        .value("maxwell_sigma", GeneratorType::maxwell_sigma)
        .value("maxwell_boltzmann", GeneratorType::maxwell_boltzmann);
}

}
