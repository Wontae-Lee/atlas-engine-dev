#include "searcher_test_utils.h"

#include <atlas/searcher/kdtree_searcher.h>

#include <testkit/testkit.h>

#include <stdexcept>

namespace {

using atlas::KdTreeSearcher;
using atlas::SearcherHostPtr;
using atlas::Vector3I;
using atlas::test::searcher::contains_neighbor;
using atlas::test::searcher::make_axis_pruning_fluid;
using atlas::test::searcher::make_fluid;
using atlas::test::searcher::make_neighbor_fluid;
using atlas::test::searcher::make_universe;
using atlas::test::searcher::valid_neighbor_slots;
using atlas::test::vec_near;

} // namespace

TEST(KdTreeSearcher, BuilderConstructsUsableSearcher) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    const auto searcher = KdTreeSearcher<float>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build();

    EXPECT_TRUE(vec_near(searcher.grid_size(), Vector3I(3, 3, 3), 0));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
}

TEST(KdTreeSearcher, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(
        KdTreeSearcher<float>::builder()
            .with_universe(universe)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        KdTreeSearcher<float>::builder()
            .with_fluid(fluid)
            .build(),
        std::invalid_argument);
}

TEST(KdTreeSearcher, BuildsThroughAbstractSearcherInterface) {
    const auto universe = make_universe();
    const auto fluid = make_neighbor_fluid();

    SearcherHostPtr<float> searcher = KdTreeSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    ASSERT_NE(searcher, nullptr);
    EXPECT_NO_THROW(searcher->build());
    EXPECT_TRUE(vec_near(searcher->grid_size(), Vector3I(3, 3, 3), 0));
    EXPECT_NE(searcher->cell_start(), nullptr);
    EXPECT_NE(searcher->cell_end(), nullptr);
}

TEST(KdTreeSearcher, BuildCreatesNeighborSlotsForNearbyParticles) {
    const auto universe = make_universe();
    const auto fluid = make_neighbor_fluid();

    SearcherHostPtr<float> searcher = KdTreeSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    searcher->build();

    EXPECT_EQ(searcher->neighbor_count(), 16);
    EXPECT_TRUE(contains_neighbor(searcher, 0, 1));
    EXPECT_TRUE(contains_neighbor(searcher, 1, 0));
    EXPECT_TRUE(contains_neighbor(searcher, 2, 3));
    EXPECT_TRUE(contains_neighbor(searcher, 3, 2));
    EXPECT_FALSE(contains_neighbor(searcher, 0, 2));
}

TEST(KdTreeSearcher, AxisPruningRejectsCandidatesOutsideSearchRadius) {
    const auto universe = make_universe();
    const auto fluid = make_axis_pruning_fluid();

    SearcherHostPtr<float> searcher = KdTreeSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    searcher->build();

    EXPECT_TRUE(contains_neighbor(searcher, 0, 1));
    EXPECT_TRUE(contains_neighbor(searcher, 1, 0));
    EXPECT_TRUE(contains_neighbor(searcher, 1, 2));
    EXPECT_TRUE(contains_neighbor(searcher, 2, 1));
    EXPECT_FALSE(contains_neighbor(searcher, 0, 2));
    EXPECT_FALSE(contains_neighbor(searcher, 2, 0));
    EXPECT_EQ(valid_neighbor_slots(searcher, 0), 1);
}

TEST(KdTreeSearcher, EmptyFluidBuildClearsNeighborState) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    SearcherHostPtr<float> searcher = KdTreeSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    searcher->build();

    EXPECT_EQ(searcher->neighbor_count(), 0);
    EXPECT_NE(searcher->cell_start(), nullptr);
    EXPECT_NE(searcher->cell_end(), nullptr);
}
