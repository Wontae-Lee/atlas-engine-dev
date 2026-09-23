#pragma once

#include <atlas/solver/solver.h>

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_solver_type(nb::module_& m) {
    nb::enum_<SolverType>(m, "SolverType")
        .value("dsmc", SolverType::dsmc);
}

}
