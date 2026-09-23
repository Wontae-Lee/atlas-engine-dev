#pragma once

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/math/vector/float3.h>

#include <nanobind/nanobind.h>

#include <cstddef>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_generator(nb::module_& m) {
    nb::class_<Generator>(m, "Generator")
        .def_ro("type", &Generator::type)
        .def("set_bulk_velocity", &Generator::set_bulk_velocity, "bulk_velocity"_a)
        .def_prop_rw(
            "bulk_velocity",
            [](const Generator& generator) {
                return GeneratorVariant::visit(
                    generator,
                    [](const auto& leaf) {
                        return leaf.bulk_velocity();
                    },
                    Float3 {});
            },
            &Generator::set_bulk_velocity)
        .def_prop_ro("temperature", [](const Generator& generator) {
            return GeneratorVariant::visit(
                generator,
                [](const auto& leaf) {
                    return leaf.temperature();
                },
                0.0f);
        })
        .def(
            "generate",
            [](const Generator& generator, Fluid& fluid, const std::size_t offset, const std::size_t count) {
                return generator.generate(fluid.state<FluidVelocityState>(),
                                          fluid.state<FluidSpeciesState>(),
                                          offset,
                                          count);
            },
            "fluid"_a,
            "offset"_a,
            "count"_a,
            "Writes velocity/species into a range clamped to buffer capacity and returns "
            "the written count; particle_count is unchanged.");
}

}
