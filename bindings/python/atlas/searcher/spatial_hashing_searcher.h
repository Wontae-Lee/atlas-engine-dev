#pragma once

#include "../detail/array.h"

#include <atlas/fluid/fluid.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <nanobind/nanobind.h>

#include <cstddef>
#include <limits>

namespace nb = nanobind;
using namespace nb::literals;

namespace atlas::python {

inline void
require_matching_grid(const SpatialHashingSearcher& searcher, const Universe& universe) {
    if (universe.cell_size() != searcher.cell_size()
        || universe.lower_corner() != searcher.lower_corner()
        || universe.grid_size() != searcher.grid_size()) {
        throw nb::value_error("universe geometry must match the searcher grid");
    }
}

inline void
register_spatial_hashing_searcher(nb::module_& m) {
    auto type = nb::class_<SpatialHashingSearcher>(m, "SpatialHashingSearcher")
        .def(
            "classify",
            [](SpatialHashingSearcher& searcher, const Fluid& fluid, Universe* universe) {
                const std::size_t count = fluid.particle_count();
                if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw nb::value_error("particle_count exceeds the searcher's integer range");
                }
                if (universe != nullptr) require_matching_grid(searcher, *universe);
                const auto* positions = fluid.state<FluidPositionState>();
                auto* counts          = universe ? universe->state<UniverseNumberParticleState>() : nullptr;
                searcher.classify(positions, counts, static_cast<int>(count));
            },
            "fluid"_a,
            "universe"_a.none() = nb::none(),
            "Classify live particles, optionally publishing counts into a matching universe.")
        .def("reset", &SpatialHashingSearcher::reset)
        .def_prop_ro("particle_count", &SpatialHashingSearcher::particle_count)
        .def_prop_ro("lower_corner", &SpatialHashingSearcher::lower_corner)
        .def_prop_ro("grid_size", &SpatialHashingSearcher::grid_size)
        .def_prop_ro("cell_count", &SpatialHashingSearcher::cell_count)
        .def_prop_ro("cell_size", &SpatialHashingSearcher::cell_size)
        .def_prop_ro("inverse_cell_size", &SpatialHashingSearcher::inverse_cell_size)
        .def("cell_key", [](const SpatialHashingSearcher& searcher) {
            return numpy_copy_device(searcher.cell_key(), searcher.particle_count());
        })
        .def("indices", [](const SpatialHashingSearcher& searcher) {
            return numpy_copy_device(searcher.indices(), searcher.particle_count());
        })
        .def("cell_start", [](const SpatialHashingSearcher& searcher) {
            return numpy_copy_device(searcher.cell_start(),
                                     static_cast<std::size_t>(searcher.cell_count()));
        })
        .def("cell_end", [](const SpatialHashingSearcher& searcher) {
            return numpy_copy_device(searcher.cell_end(),
                                     static_cast<std::size_t>(searcher.cell_count()));
        })
        .def(
            "cell_for",
            [](const SpatialHashingSearcher& searcher, const Float3& position) {
                return SpatialHashingSearcher::cell_for(position,
                                                        searcher.lower_corner(),
                                                        searcher.inverse_cell_size(),
                                                        searcher.grid_size());
            },
            "position"_a)
        .def(
            "linear_key",
            [](const SpatialHashingSearcher& searcher, const Int3& cell) {
                if (!SpatialHashingSearcher::contains_cell(cell, searcher.grid_size())) {
                    throw nb::value_error("cell coordinates must lie inside the searcher grid");
                }
                return SpatialHashingSearcher::linear_key(cell, searcher.grid_size());
            },
            "cell"_a)
        .def(
            "contains_cell",
            [](const SpatialHashingSearcher& searcher, const Int3& cell) {
                return SpatialHashingSearcher::contains_cell(cell, searcher.grid_size());
            },
            "cell"_a);

    type.def(nb::new_([](const Universe& universe) {
            return SpatialHashingSearcher::builder().with_universe(universe).build();
        }),
        "universe"_a,
        "Build an empty spatial hash using the universe's grid geometry.");

    type.def(nb::new_([](const Float3& lower_corner, const float cell_size, const Int3& grid_size) {
            return SpatialHashingSearcher::builder()
                .with_lower_corner(lower_corner)
                .with_cell_size(cell_size)
                .with_grid_size(grid_size)
                .build();
        }),
        "lower_corner"_a,
        "cell_size"_a,
        "grid_size"_a,
        "Build an empty spatial hash with explicit grid geometry.");
}

}
