#pragma once

#include <atlas/solver/solver.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_solver(nb::module_& m) {
    nb::class_<Solver>(m, "Solver")
        .def_prop_ro("type", &Solver::type);
}

}
