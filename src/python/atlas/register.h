#pragma once

#include <nanobind/nanobind.h>

// Each translation unit under src/python/atlas binds one slice of the engine
// into the shared `atlas` module. module.cpp owns NB_MODULE and calls every
// register_* hook below in dependency order (math and geometry first, since the
// assembly types take Float3 and Geometry by value).
namespace atlas::python {

void register_math(nanobind::module_& m);
void register_geometry(nanobind::module_& m);
void register_material(nanobind::module_& m);
void register_transform(nanobind::module_& m);
void register_fluid_universe(nanobind::module_& m);
void register_emitter(nanobind::module_& m);
void register_solver(nanobind::module_& m);
void register_boundary(nanobind::module_& m);
void register_system(nanobind::module_& m);

}
