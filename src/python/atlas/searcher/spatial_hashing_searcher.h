#pragma once

#include "../_detail/searcher.h"

namespace atlas::python {

inline void
register_spatial_hashing_searcher(nb::module_& m) {
    auto type = nb::class_<PySearcher>(m, "SpatialHashingSearcher")
        .def(
            "classify",
            [](PySearcher& searcher, const Fluid& fluid, Universe* universe) {
                const std::size_t count = fluid.particle_count();
                if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw nb::value_error("particle_count exceeds the searcher's integer range");
                }
                if (universe != nullptr) require_matching_grid(searcher.value, *universe);
                const auto* positions = fluid.state<FluidPositionState>();
                auto* counts          = universe ? universe->state<UniverseNumberParticleState>() : nullptr;
                searcher.value.classify(positions, counts, static_cast<int>(count));
                searcher.particle_count = positions != nullptr && count <= positions->size() ? count : 0;
            },
            "fluid"_a,
            "universe"_a.none() = nb::none(),
            "Classify live particles, optionally publishing counts into a matching universe.")
        .def(
            "solve",
            [](const PySearcher& searcher, Solver& solver, Fluid& fluid, Universe& universe, const int index, const float dt) {
                require_matching_grid(searcher.value, universe);
                if (fluid.particle_count() != searcher.particle_count) {
                    throw nb::value_error("classify the current particle population before solving");
                }
                solver.solve(fluid, universe, searcher.value.view(), index, dt);
            },
            "solver"_a,
            "fluid"_a,
            "universe"_a,
            "index"_a,
            "dt"_a,
            "Run a solver using this index. Provision the solver's universe states first; "
            "classify again after changing particle positions, order, or population.")
        .def("reset", [](PySearcher& searcher) {
            searcher.value.reset();
            searcher.particle_count = 0;
        })
        .def_prop_ro("particle_count", [](const PySearcher& searcher) {
            return searcher.particle_count;
        })
        .def_prop_ro("lower_corner", [](const PySearcher& searcher) {
            return searcher.value.lower_corner();
        })
        .def_prop_ro("grid_size", [](const PySearcher& searcher) {
            return searcher.value.grid_size();
        })
        .def_prop_ro("cell_count", [](const PySearcher& searcher) {
            return searcher.value.cell_count();
        })
        .def_prop_ro("cell_size", [](const PySearcher& searcher) {
            return searcher.value.cell_size();
        })
        .def_prop_ro("inverse_cell_size", [](const PySearcher& searcher) {
            return searcher.value.inverse_cell_size();
        })
        .def("cell_key", [](const PySearcher& searcher) {
            return searcher_array(searcher.value.cell_key(), searcher.particle_count);
        })
        .def("indices", [](const PySearcher& searcher) {
            return searcher_array(searcher.value.indices(), searcher.particle_count);
        })
        .def("cell_start", [](const PySearcher& searcher) {
            return searcher_array(searcher.value.cell_start(),
                                  static_cast<std::size_t>(searcher.value.cell_count()));
        })
        .def("cell_end", [](const PySearcher& searcher) {
            return searcher_array(searcher.value.cell_end(),
                                  static_cast<std::size_t>(searcher.value.cell_count()));
        })
        .def(
            "cell_for",
            [](const PySearcher& searcher, const Float3& position) {
                return SpatialHashingSearcher::cell_for(position,
                                                        searcher.value.lower_corner(),
                                                        searcher.value.inverse_cell_size(),
                                                        searcher.value.grid_size());
            },
            "position"_a)
        .def(
            "linear_key",
            [](const PySearcher& searcher, const Int3& cell) {
                if (!SpatialHashingSearcher::contains_cell(cell, searcher.value.grid_size())) {
                    throw nb::value_error("cell coordinates must lie inside the searcher grid");
                }
                return SpatialHashingSearcher::linear_key(cell, searcher.value.grid_size());
            },
            "cell"_a)
        .def(
            "contains_cell",
            [](const PySearcher& searcher, const Int3& cell) {
                return SpatialHashingSearcher::contains_cell(cell, searcher.value.grid_size());
            },
            "cell"_a);

    type.def(nb::new_([](const Universe& universe) {
            return PySearcher(SpatialHashingSearcher::builder().with_universe(universe).build());
        }),
        "universe"_a,
        "Build an empty spatial hash using the universe's grid geometry.");

    type.def(nb::new_([](const Float3& lower_corner, const float cell_size, const Int3& grid_size) {
            return PySearcher(SpatialHashingSearcher::builder()
                                  .with_lower_corner(lower_corner)
                                  .with_cell_size(cell_size)
                                  .with_grid_size(grid_size)
                                  .build());
        }),
        "lower_corner"_a,
        "cell_size"_a,
        "grid_size"_a,
        "Build an empty spatial hash with explicit grid geometry.");
}

}
