#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>

#include <atlas/system/system.h>

#include <filesystem>

namespace nb = nanobind;

using namespace nb::literals;

// The bindings were emptied when the engine was restructured: the old module
// exposed types that no longer exist (MaterialProperties, the Searcher base,
// SphSolver, the measurers). What remains is the smallest module that still
// builds, so the bindings can be grown back one type at a time against the
// current API.
NB_MODULE(atlas, m) {
    m.doc() = "Atlas Engine";

    nb::class_<atlas::System>(m, "System")
        .def("update", &atlas::System::update)
        .def("save", &atlas::System::save, "directory"_a)
        .def_prop_ro("step", &atlas::System::step)
        .def_prop_ro("dt", &atlas::System::dt);
}
