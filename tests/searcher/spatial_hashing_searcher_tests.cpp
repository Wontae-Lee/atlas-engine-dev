#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/kdtree_searcher.h>
#include <atlas/searcher/octree_searcher.h>
#include <atlas/searcher/quadtree_searcher.h>
#include <atlas/searcher/searcher.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

#include <testkit/testkit.h>

#include <cstdint>

namespace {

using atlas::FluidHostPtr;
using atlas::KdTreeSearcher;
using atlas::OctreeSearcher;
using atlas::QuadtreeSearcher;
using atlas::SearcherHostPtr;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::Vector3I;
using atlas::fluid::Fluid;
using atlas::fluid::FluidPositionState;
using atlas::system::SpatialHashingSearcher;
using atlas::test::vec_near;
using atlas::tol;
using atlas::universe::Universe;

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

FluidHostPtr<float>
make_fluid() {
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

FluidHostPtr<float>
make_neighbor_fluid() {
    auto fluid = Fluid<float>::builder()
                     .with_buffer_size(4)
                     .make_host_shared();

    fluid->set_particle_count(4);
    auto& positions = fluid->state<FluidPositionState<float>>()->data();
    positions[0] = Vector3F(0.10f, 0.10f, 0.10f);
    positions[1] = Vector3F(0.20f, 0.10f, 0.10f);
    positions[2] = Vector3F(0.85f, 0.85f, 0.85f);
    positions[3] = Vector3F(0.95f, 0.85f, 0.85f);

    return fluid;
}

bool
contains_neighbor(const SearcherHostPtr<float>& searcher, const int particle, const int neighbor) {
    const int* offsets = searcher->neighbor_offsets();
    const int* indices = searcher->neighbor_indices();

    if (!offsets || !indices) {
        return false;
    }

    for (int i = offsets[particle]; i < offsets[particle + 1]; ++i) {
        if (indices[i] == neighbor) {
            return true;
        }
    }

    return false;
}

} // namespace

TEST(SpatialHashingSearcher, BuilderConstructsUsableSearcher) {
    // Arrange: create the required universe and fluid dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    // Act: build a spatial hashing searcher.
    const auto searcher = SpatialHashingSearcher<float>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build();

    // Assert: derived spatial hashing parameters match the universe domain.
    EXPECT_TRUE(vec_near(searcher.lower_corner(), Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(searcher.grid_size(), Vector3I(3, 3, 3), 0));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
}

TEST(SpatialHashingSearcher, BuilderRejectsMissingDependencies) {
    // Arrange: create dependencies used by individual failure cases.
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    // A searcher cannot be built without a fluid.
    EXPECT_THROW(
        SpatialHashingSearcher<float>::builder()
            .with_universe(universe)
            .build(),
        std::invalid_argument);

    // A searcher cannot be built without a universe.
    EXPECT_THROW(
        SpatialHashingSearcher<float>::builder()
            .with_fluid(fluid)
            .build(),
        std::invalid_argument);
}

TEST(SpatialHashingSearcher, MakeHostSharedBuildsSearcher) {
    // Arrange: create the required universe and fluid dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    // Act: build a searcher through host shared ownership.
    const auto searcher = SpatialHashingSearcher<float>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    // Assert: the shared searcher exists and exposes derived grid data.
    ASSERT_NE(searcher, nullptr);
    EXPECT_FLOAT_EQ(searcher->cell_size(), 0.5f);
}

TEST(SpatialHashingSearcher, LinearKeyMatchesFlattenedIndexing) {
    // Act: flatten a 3D cell coordinate into a linear key.
    const auto key = SpatialHashingSearcher<float>::linear_key(1, 2, 1, Vector3I(4, 5, 6));

    // Assert: the key follows row-major flattened indexing.
    EXPECT_EQ(key, static_cast<std::uint32_t>(1 + 2 * 4 + 1 * 4 * 5));
}

TEST(SpatialHashingSearcher, ResetIsSafeWithEmptyFluid) {
    // Arrange: build a searcher over an empty active fluid prefix.
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    auto searcher = SpatialHashingSearcher<float>::builder()
                        .with_universe(universe)
                        .with_fluid(fluid)
                        .build();

    // Act and assert: reset keeps search buffers available.
    EXPECT_NO_THROW(searcher.reset());
    EXPECT_NE(searcher.cell_start(), nullptr);
    EXPECT_NE(searcher.cell_end(), nullptr);
}

TEST(Searcher, SpatialHashingSearcherCanBeUsedThroughAbstractInterface) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    SearcherHostPtr<float> searcher = SpatialHashingSearcher<float>::builder()
                                          .with_universe(universe)
                                          .with_fluid(fluid)
                                          .make_host_shared();

    ASSERT_NE(searcher, nullptr);
    EXPECT_NO_THROW(searcher->build());
    EXPECT_TRUE(vec_near(searcher->lower_corner(), Vector3F(0, 0, 0), tol));
    EXPECT_TRUE(vec_near(searcher->grid_size(), Vector3I(3, 3, 3), 0));
    EXPECT_FLOAT_EQ(searcher->cell_size(), 0.5f);
}

TEST(Searcher, TreeSearchersExposeGridCompatibleSearchView) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    const SearcherHostPtr<float> kd_tree = KdTreeSearcher<float>::builder()
                                               .with_universe(universe)
                                               .with_fluid(fluid)
                                               .make_host_shared();

    const SearcherHostPtr<float> octree = OctreeSearcher<float>::builder()
                                              .with_universe(universe)
                                              .with_fluid(fluid)
                                              .make_host_shared();

    const SearcherHostPtr<float> quadtree = QuadtreeSearcher<float>::builder()
                                                .with_universe(universe)
                                                .with_fluid(fluid)
                                                .make_host_shared();

    for (const auto& searcher : { kd_tree, octree, quadtree }) {
        ASSERT_NE(searcher, nullptr);
        EXPECT_NO_THROW(searcher->build());
        EXPECT_TRUE(vec_near(searcher->grid_size(), Vector3I(3, 3, 3), 0));
        EXPECT_NE(searcher->cell_start(), nullptr);
        EXPECT_NE(searcher->cell_end(), nullptr);
    }
}

TEST(Searcher, ConcreteSearchersBuildParticleNeighborLists) {
    const auto universe = make_universe();
    const auto fluid = make_neighbor_fluid();

    const SearcherHostPtr<float> spatial = SpatialHashingSearcher<float>::builder()
                                               .with_universe(universe)
                                               .with_fluid(fluid)
                                               .make_host_shared();

    const SearcherHostPtr<float> kd_tree = KdTreeSearcher<float>::builder()
                                               .with_universe(universe)
                                               .with_fluid(fluid)
                                               .make_host_shared();

    const SearcherHostPtr<float> octree = OctreeSearcher<float>::builder()
                                              .with_universe(universe)
                                              .with_fluid(fluid)
                                              .make_host_shared();

    const SearcherHostPtr<float> quadtree = QuadtreeSearcher<float>::builder()
                                                .with_universe(universe)
                                                .with_fluid(fluid)
                                                .make_host_shared();

    for (const auto& searcher : { spatial, kd_tree, octree, quadtree }) {
        searcher->build();
        EXPECT_GT(searcher->neighbor_count(), 0);
        EXPECT_TRUE(contains_neighbor(searcher, 0, 1));
        EXPECT_TRUE(contains_neighbor(searcher, 2, 3));
    }
}
