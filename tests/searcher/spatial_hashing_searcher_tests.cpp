#include "searcher_test_utils.h"

#include <atlas/searcher/spatial_hashing_searcher.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::Float3;
using atlas::Int3;
using atlas::test::searcher::contains_neighbor;
using atlas::test::searcher::copy_values;
using atlas::test::searcher::expect_vec_near;
using atlas::test::searcher::make_fluid;
using atlas::test::searcher::make_neighbor_fluid;
using atlas::test::searcher::make_universe;
using atlas::test::searcher::valid_neighbor_slots;

}

TEST(SpatialHashingSearcher, BuilderConstructsUsableSearcher) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    const auto searcher = SpatialHashingSearcher::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build();

    EXPECT_TRUE(expect_vec_near(searcher.lower_corner(), Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(searcher.grid_size() == Int3(3, 3, 3));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
}

TEST(SpatialHashingSearcher, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    EXPECT_THROW(
        static_cast<void>(SpatialHashingSearcher::builder()
                              .with_universe(universe)
                              .build()),
        std::invalid_argument);

    EXPECT_THROW(
        static_cast<void>(SpatialHashingSearcher::builder()
                              .with_fluid(fluid)
                              .build()),
        std::invalid_argument);
}

TEST(SpatialHashingSearcher, MakeHostSharedBuildsAbstractCompatibleSearcher) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    SearcherHostPtr searcher = SpatialHashingSearcher::builder()
                                   .with_universe(universe)
                                   .with_fluid(fluid)
                                   .make_host_shared();

    ASSERT_NE(searcher, nullptr);
    EXPECT_NO_THROW(searcher->build());
    EXPECT_FLOAT_EQ(searcher->cell_size(), 0.5f);
    EXPECT_TRUE(searcher->grid_size() == Int3(3, 3, 3));
}

TEST(SpatialHashingSearcher, LinearKeyForwardsBaseFlattening) {
    const auto key = SpatialHashingSearcher::linear_key(1, 2, 1, Int3(4, 5, 6));
    EXPECT_EQ(key, static_cast<std::uint32_t>(1 + 2 * 4 + 1 * 4 * 5));
}

TEST(SpatialHashingSearcher, ResetIsSafeWithEmptyFluid) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    auto searcher = SpatialHashingSearcher::builder()
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
    const auto fluid    = make_neighbor_fluid();

    SearcherHostPtr searcher = SpatialHashingSearcher::builder()
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

    const std::size_t cell_count = static_cast<std::size_t>(universe->cell_count());
    const std::vector<int> starts  = copy_values(searcher->cell_start(), cell_count);
    const std::vector<int> ends    = copy_values(searcher->cell_end(), cell_count);

    EXPECT_GE(starts[0], 0);
    EXPECT_GT(ends[0], starts[0]);
}

TEST(SpatialHashingSearcher, BuildIsSkippedUntilInvalidated) {
    const auto universe = make_universe();
    const auto fluid    = make_neighbor_fluid();

    SearcherHostPtr searcher = SpatialHashingSearcher::builder()
                                   .with_universe(universe)
                                   .with_fluid(fluid)
                                   .make_host_shared();

    searcher->build();
    ASSERT_TRUE(contains_neighbor(searcher, 0, 1));

    auto& positions = fluid->state<atlas::FluidPositionState>()->data();
    positions[1]    = Float3(0.90f, 0.90f, 0.90f);

    searcher->build();
    EXPECT_TRUE(contains_neighbor(searcher, 0, 1));

    searcher->invalidate();
    searcher->build();
    EXPECT_FALSE(contains_neighbor(searcher, 0, 1));
}
