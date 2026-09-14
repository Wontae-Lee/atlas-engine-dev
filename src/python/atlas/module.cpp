#include "register.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// The engine bindings are split across bindings_*.cpp, one register_* hook per
// module group (see register.h). They run in dependency order: math and
// geometry define the value types (Float3, Quaternion, Geometry) the later
// groups take as arguments, and the system group ties everything into a runnable
// System at the end.
NB_MODULE(atlas, m) {
    m.doc() = "Atlas Engine — GPU/CPU rarefied-gas (DSMC) particle simulation.";
    m.attr("__version__") = ATLAS_VERSION_STRING;

    atlas::python::register_math(m);
    atlas::python::register_sampling(m);
    atlas::python::register_geometry(m);
    atlas::python::register_bvh(m);
    atlas::python::register_transform(m);
    atlas::python::register_material(m);
    atlas::python::register_fluid_universe(m);
    atlas::python::register_emitter(m);
    atlas::python::register_solver(m);
    atlas::python::register_boundary(m);
    atlas::python::register_system(m);
    atlas::python::register_searcher(m);
    atlas::python::register_serialization(m);
}
