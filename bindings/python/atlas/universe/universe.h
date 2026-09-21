#pragma once

#include "../detail/ownership.h"
#include "universe_state.h"

#include <atlas/geometry/geometry.h>
#include <atlas/math/vector/float3.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
register_universe(nb::module_& m) {
    auto type = nb::class_<Universe>(m, "Universe")
        .def_prop_ro("cell_count", &Universe::cell_count)
        .def_prop_ro("lower_corner", &Universe::lower_corner)
        .def_prop_ro("upper_corner", &Universe::upper_corner)
        .def_prop_ro("grid_size", &Universe::grid_size)
        .def_prop_ro("cell_size", &Universe::cell_size)
        .def_prop_ro("cell_volume", &Universe::cell_volume)
        .def_prop_ro("inverse_cell_size", &Universe::inverse_cell_size)
        .def(
            "state",
            [](const Universe& universe, const std::string& name) {
                return read_universe_state(universe, name);
            },
            "name"_a)
        .def("set_state", &write_universe_state, "name"_a, "values"_a, "offset"_a = 0)
        .def("has_state", &has_universe_state, "name"_a)
        .def("remove_state", &remove_universe_state, "name"_a)
        .def("reset_state", &reset_universe_state, "name"_a);

    type.def(nb::new_([](const Float3& lower, const Float3& upper, const float cell_size) {
            return Universe::builder()
                .with_lower_corner(lower)
                .with_upper_corner(upper)
                .with_cell_size(cell_size)
                .make_host_unique();
        }),
        "lower"_a,
        "upper"_a,
        "cell_size"_a,
        "A uniform grid spanning [lower, upper] with cubic cells of the given size.");

    type.def_static("from_geometry",
        [](const PyGeometry& geometry, const float cell_size) {
            return Universe::builder()
                .with_geometry(geometry.value)
                .with_cell_size(cell_size)
                .make_host_unique();
        },
        "geometry"_a,
        "cell_size"_a,
        "A uniform grid whose domain is taken from a geometry's bounding box.");

    type.def_static("load",
        [](const std::filesystem::path& path) { return restore_universe(path.string()); },
        "path"_a,
        "Rebuild a Universe from a binary snapshot file.");
}

}
