#include "searcher_test_utils.h"

#include <atlas/searcher/spatial_hashing_searcher.h>

#include <testkit/testkit.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using atlas::SearcherHostPtr;
using atlas::Vector3F;
using atlas::Vector3I;
using atlas::system::SpatialHashingSearcher;
using atlas::test::searcher::contains_neighbor;
using atlas::test::searcher::copy_values;
using atlas::test::searcher::make_fluid;
using atlas::test::searcher::make_neighbor_fluid;
using atlas::test::searcher::make_universe;
using atlas::test::searcher::valid_neighbor_slots;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(SpatialHashingSearcher, BuilderConstructsUsableSearcher) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    const auto searcher = SpatialHashingSearcher<float>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build();

    EXPECT_TRUE(vec_near(searcher.lower_corner(), Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(searcher.grid_size(), Vector3I(3, 3, 3), 0));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
}

TEST(SpatialHashingSearcher, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(
        SpatialHashingSearcher<float>::builder()
            .with_universe(universe)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        SpatialHashingSearcher<float>::builder()
            .with_fluid(fluid)
            .build(),
        std::invalid_argument);
}

TEST(SpatialHashingSearcher, MakeHostSharedBuildsAbstractCompatibleSearcher) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    SearcherHostPtr<float> searcher = SpatialHashingSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    ASSERT_NE(searcher, nullptr);
    EXPECT_NO_THROW(searcher->build());
    EXPECT_FLOAT_EQ(searcher->cell_size(), 0.5f);
    EXPECT_TRUE(vec_near(searcher->grid_size(), Vector3I(3, 3, 3), 0));
}

TEST(SpatialHashingSearcher, LinearKeyForwardsBaseFlattening) {
    const auto key = SpatialHashingSearcher<float>::linear_key(1, 2, 1, Vector3I(4, 5, 6));
    EXPECT_EQ(key, static_cast<std::uint32_t>(1 + 2 * 4 + 1 * 4 * 5));
}

TEST(SpatialHashingSearcher, ResetIsSafeWithEmptyFluid) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    auto searcher = SpatialHashingSearcher<float>::builder()
                        .with_universe(universe)
                        .with_fluid(fluid)
                        .build();

    EXPECT_NO_THROW(searcher.reset());
    EXPECT_NE(searcher.cell_start(), nullptr);
    EXPECT_NE(searcher.cell_end(), nullptr);
    EXPECT_EQ(searcher.neighbor_count(), 0);
}

TEST(SpatialHashingSearcher, BuildCreatesGridViewAndNeighborSlots) {
    const auto universe = make_universe();
    const auto fluid = make_neighbor_fluid();

    SearcherHostPtr<float> searcher = SpatialHashingSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    searcher->build();

    EXPECT_EQ(searcher->neighbor_count(), 4);
    EXPECT_TRUE(contains_neighbor(searcher, 0, 1));
    EXPECT_TRUE(contains_neighbor(searcher, 1, 0));
    EXPECT_TRUE(contains_neighbor(searcher, 2, 3));
    EXPECT_TRUE(contains_neighbor(searcher, 3, 2));
    EXPECT_EQ(valid_neighbor_slots(searcher, 0), 1);
    EXPECT_EQ(valid_neighbor_slots(searcher, 2), 1);

    const std::vector<int> starts = copy_values(searcher->cell_start(), universe->number_of_cells());
    const std::vector<int> ends = copy_values(searcher->cell_end(), universe->number_of_cells());

    EXPECT_GE(starts[0], 0);
    EXPECT_GT(ends[0], starts[0]);
}

TEST(SpatialHashingSearcher, BuildIsSkippedUntilInvalidated) {
    const auto universe = make_universe();
    const auto fluid = make_neighbor_fluid();

    SearcherHostPtr<float> searcher = SpatialHashingSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    searcher->build();
    ASSERT_TRUE(contains_neighbor(searcher, 0, 1));

    auto& positions = fluid->state<atlas::fluid::FluidPositionState<float>>()->data();
    positions[1] = Vector3F(0.90f, 0.90f, 0.90f);

    searcher->build();
    EXPECT_TRUE(contains_neighbor(searcher, 0, 1));

    searcher->invalidate();
    searcher->build();
    EXPECT_FALSE(contains_neighbor(searcher, 0, 1));
}
