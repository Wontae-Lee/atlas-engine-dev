#include "searcher_test_utils.h"

#include <atlas/searcher/quadtree_searcher.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::QuadtreeSearcher;
using atlas::SearcherHostPtr;
using atlas::Vector3i;
using atlas::test::searcher::contains_neighbor;
using atlas::test::searcher::make_fluid;
using atlas::test::searcher::make_quadrant_fluid;
using atlas::test::searcher::make_universe;
using atlas::test::searcher::valid_neighbor_slots;

}

TEST(QuadtreeSearcher, BuilderConstructsUsableSearcher) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    const auto searcher = QuadtreeSearcher::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build();

    EXPECT_TRUE(searcher.grid_size() == Vector3i(3, 3, 3));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
}

TEST(QuadtreeSearcher, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    EXPECT_THROW(
        static_cast<void>(QuadtreeSearcher::builder()
                              .with_universe(universe)
                              .build()),
        std::invalid_argument);

    EXPECT_THROW(
        static_cast<void>(QuadtreeSearcher::builder()
                              .with_fluid(fluid)
                              .build()),
        std::invalid_argument);
}

TEST(QuadtreeSearcher, BuildsThroughAbstractSearcherInterface) {
    const auto universe = make_universe();
    const auto fluid    = make_quadrant_fluid();

    SearcherHostPtr searcher = QuadtreeSearcher::builder()
                                   .with_universe(universe)
                                   .with_fluid(fluid)
                                   .make_host_shared();

    ASSERT_NE(searcher, nullptr);
    EXPECT_NO_THROW(searcher->build());
    EXPECT_EQ(searcher->neighbor_count(), 4);
    EXPECT_TRUE(searcher->grid_size() == Vector3i(3, 3, 3));
}

TEST(QuadtreeSearcher, SameQuadrantParticlesBecomeNeighbors) {
    const auto universe = make_universe();
    const auto fluid    = make_quadrant_fluid();

    SearcherHostPtr searcher = QuadtreeSearcher::builder()
                                   .with_universe(universe)
                                   .with_fluid(fluid)
                                   .make_host_shared();

    searcher->build();

    EXPECT_TRUE(contains_neighbor(searcher, 0, 1));
    EXPECT_TRUE(contains_neighbor(searcher, 1, 0));
    EXPECT_TRUE(contains_neighbor(searcher, 2, 3));
    EXPECT_TRUE(contains_neighbor(searcher, 3, 2));
    EXPECT_EQ(valid_neighbor_slots(searcher, 0), 1);
    EXPECT_EQ(valid_neighbor_slots(searcher, 2), 1);
}

TEST(QuadtreeSearcher, DifferentQuadrantsAreRejectedBeforeDistanceAcceptance) {
    const auto universe = make_universe();
    const auto fluid    = make_quadrant_fluid();

    SearcherHostPtr searcher = QuadtreeSearcher::builder()
                                   .with_universe(universe)
                                   .with_fluid(fluid)
                                   .make_host_shared();

    searcher->build();

    EXPECT_FALSE(contains_neighbor(searcher, 0, 2));
    EXPECT_FALSE(contains_neighbor(searcher, 0, 3));
    EXPECT_FALSE(contains_neighbor(searcher, 1, 2));
    EXPECT_FALSE(contains_neighbor(searcher, 1, 3));
}

TEST(QuadtreeSearcher, EmptyFluidBuildClearsNeighborState) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();

    SearcherHostPtr searcher = QuadtreeSearcher::builder()
                                   .with_universe(universe)
                                   .with_fluid(fluid)
                                   .make_host_shared();

    searcher->build();

    EXPECT_EQ(searcher->neighbor_count(), 0);
    EXPECT_NE(searcher->cell_start(), nullptr);
    EXPECT_NE(searcher->cell_end(), nullptr);
}
