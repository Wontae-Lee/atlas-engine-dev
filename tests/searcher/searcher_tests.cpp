#include "searcher_test_utils.h"

#include <testkit/testkit.h>

#include <stdexcept>
#include <vector>

namespace {

using atlas::FluidHostPtr;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::Vector3I;
using atlas::Searcher;
using atlas::test::searcher::ExposedSearcher;
using atlas::test::searcher::copy_values;
using atlas::test::searcher::make_fluid;
using atlas::test::searcher::make_grid_fluid;
using atlas::test::searcher::make_universe;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Searcher, DefaultConstructedInheritedStateHasSafeFallbacks) {
    ExposedSearcher searcher;

    EXPECT_TRUE(vec_near(searcher.lower_corner(), Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(searcher.grid_size(), Vector3I(0, 0, 0), 0));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 1.0f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 1.0f);
    EXPECT_EQ(searcher.neighbor_count(), 0);

    EXPECT_NO_THROW(searcher.reset());
    EXPECT_EQ(searcher.key_size(), 0u);
    EXPECT_EQ(searcher.index_size(), 0u);
    EXPECT_EQ(searcher.cell_range_size(), 0u);
    EXPECT_TRUE(searcher.invalidated());
}

TEST(Searcher, InheritedConstructorRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(ExposedSearcher(UniverseHostPtr<float> {}, fluid), std::invalid_argument);
    EXPECT_THROW(ExposedSearcher(universe, FluidHostPtr<float> {}), std::invalid_argument);
}

TEST(Searcher, LinearKeyUsesRowMajorCellFlattening) {
    EXPECT_EQ(Searcher<float>::linear_key(0, 0, 0, Vector3I(4, 5, 6)), 0u);
    EXPECT_EQ(Searcher<float>::linear_key(1, 0, 0, Vector3I(4, 5, 6)), 1u);
    EXPECT_EQ(Searcher<float>::linear_key(0, 1, 0, Vector3I(4, 5, 6)), 4u);
    EXPECT_EQ(Searcher<float>::linear_key(0, 0, 1, Vector3I(4, 5, 6)), 20u);
    EXPECT_EQ(Searcher<float>::linear_key(3, 4, 5, Vector3I(4, 5, 6)), 119u);
}

TEST(Searcher, InheritedResetClearsParticleAndNeighborBuffers) {
    const auto universe = make_universe();
    const auto fluid = make_grid_fluid();
    ExposedSearcher searcher(universe, fluid);

    searcher.build();
    searcher.seed_neighbors_for_reset();

    ASSERT_EQ(searcher.key_size(), 4u);
    ASSERT_EQ(searcher.index_size(), 4u);
    ASSERT_EQ(searcher.neighbor_count(), 3);

    searcher.reset();

    EXPECT_EQ(searcher.key_size(), 0u);
    EXPECT_EQ(searcher.index_size(), 0u);
    EXPECT_EQ(searcher.neighbor_offset_size(), 0u);
    EXPECT_EQ(searcher.neighbor_index_size(), 0u);
    EXPECT_EQ(searcher.neighbor_count(), 0);
    EXPECT_EQ(searcher.cell_range_size(), universe->number_of_cells());
    EXPECT_TRUE(searcher.invalidated());
}

TEST(Searcher, InheritedBuildResetsEmptyActiveFluid) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    ExposedSearcher searcher(universe, fluid);

    searcher.build();

    EXPECT_EQ(searcher.key_size(), 0u);
    EXPECT_EQ(searcher.index_size(), 0u);
    EXPECT_EQ(searcher.neighbor_count(), 0);
    EXPECT_EQ(searcher.cell_range_size(), universe->number_of_cells());
    EXPECT_TRUE(searcher.invalidated());
}

TEST(Searcher, InheritedGridBuildSortsIndicesAndBuildsCellRanges) {
    const auto universe = make_universe();
    const auto fluid = make_grid_fluid();
    ExposedSearcher searcher(universe, fluid);

    searcher.build();

    const std::vector<int> indices = copy_values(searcher.indices(), searcher.index_size());
    EXPECT_EQ(indices, (std::vector<int> { 0, 1, 3, 2 }));

    const std::vector<int> starts = copy_values(searcher.cell_start(), universe->number_of_cells());
    const std::vector<int> ends = copy_values(searcher.cell_end(), universe->number_of_cells());

    const auto cell0 = Searcher<float>::linear_key(0, 0, 0, universe->grid_size());
    const auto cell1 = Searcher<float>::linear_key(1, 0, 0, universe->grid_size());
    const auto cell3 = Searcher<float>::linear_key(0, 1, 0, universe->grid_size());
    const auto cell26 = Searcher<float>::linear_key(2, 2, 2, universe->grid_size());
    const auto empty_cell = Searcher<float>::linear_key(2, 0, 0, universe->grid_size());

    EXPECT_EQ(starts[cell0], 0);
    EXPECT_EQ(ends[cell0], 1);
    EXPECT_EQ(starts[cell1], 1);
    EXPECT_EQ(ends[cell1], 2);
    EXPECT_EQ(starts[cell3], 2);
    EXPECT_EQ(ends[cell3], 3);
    EXPECT_EQ(starts[cell26], 3);
    EXPECT_EQ(ends[cell26], 4);
    EXPECT_EQ(starts[empty_cell], -1);
    EXPECT_EQ(ends[empty_cell], -1);
    EXPECT_FALSE(searcher.invalidated());
}
