#include "register.h"

#include <atlas/fluid/fluid.h>
#include <atlas/geometry/geometry.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/math/vector/float3.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cstddef>
#include <memory>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

// The two owner types that hold all simulation state: `Fluid` is the Lagrangian
// particle population and `Universe` is the Eulerian background grid. Both are
// move-only (they own device buffers) and are handed to Python through owning
// `host_unique_ptr`s built by their fluent builders, so we expose read-only
// accessors on the classes and free factory functions that run the builders.
namespace atlas::python {

void
register_fluid_universe(nb::module_& m) {
    nb::class_<Fluid>(m, "Fluid")
        .def_prop_ro("particle_count", &Fluid::particle_count)
        .def_prop_ro("buffer_size", &Fluid::buffer_size);

    m.def(
        "fluid",
        [](const std::size_t buffer_size,
           const std::size_t particle_count,
           const float statistical_weight,
           std::shared_ptr<MaterialDictionary> materials) {
            auto builder = Fluid::builder()
                               .with_buffer_size(buffer_size)
                               .with_particle_count(particle_count)
                               .with_statistical_weight(statistical_weight);
            if (materials) {
                builder.with_materials(materials);
            }
            return builder.make_host_unique();
        },
        "buffer_size"_a, "particle_count"_a = 0, "statistical_weight"_a = 1.0f,
        "materials"_a = std::shared_ptr<MaterialDictionary>(),
        "A particle population of the given capacity, optionally attached to a "
        "material dictionary.");

    nb::class_<Universe>(m, "Universe").def_prop_ro("cell_count", &Universe::cell_count);

    m.def(
        "universe",
        [](const Float3& lower, const Float3& upper, const float cell_size) {
            return Universe::builder()
                .with_lower_corner(lower)
                .with_upper_corner(upper)
                .with_cell_size(cell_size)
                .make_host_unique();
        },
        "lower"_a, "upper"_a, "cell_size"_a,
        "A uniform grid spanning [lower, upper] with cubic cells of the given size.");

    m.def(
        "universe_from_geometry",
        [](const Geometry& geometry, const float cell_size) {
            return Universe::builder()
                .with_geometry(geometry)
                .with_cell_size(cell_size)
                .make_host_unique();
        },
        "geometry"_a, "cell_size"_a,
        "A uniform grid whose domain is taken from a geometry's bounding box.");

    // Reload a snapshot previously written by System.save(): restore_* wraps the
    // binary loader plus the builder, returning a device-resident owner.
    m.def(
        "load_fluid",
        [](const std::string& path) { return restore_fluid(path); },
        "path"_a,
        "Rebuild a Fluid from a binary snapshot file.");

    m.def(
        "load_universe",
        [](const std::string& path) { return restore_universe(path); },
        "path"_a,
        "Rebuild a Universe from a binary snapshot file.");
}

}
