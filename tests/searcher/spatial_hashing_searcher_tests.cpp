#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using IVec3 = atlas::Vector3<int>;

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(Vec3(0, 0, 0))
        .with_upper_corner(Vec3(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

} // namespace

TEST(SpatialHashingSearcher, BuilderConstructsUsableSearcher) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    const auto searcher = atlas::system::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build();

    EXPECT_TRUE(atlas::test::vec_near(searcher.lower_corner(), Vec3(0, 0, 0), 1e-6f));
    EXPECT_TRUE(atlas::test::vec_near(searcher.grid_size(), IVec3(3, 3, 3), 0));
    EXPECT_FLOAT_EQ(searcher.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(searcher.inverse_cell_size(), 2.0f);
}

TEST(SpatialHashingSearcher, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::system::SpatialHashingSearcher<T>::builder()
            .with_universe(universe)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::system::SpatialHashingSearcher<T>::builder()
            .with_fluid(fluid)
            .build(),
        std::invalid_argument);
}

TEST(SpatialHashingSearcher, MakeHostSharedBuildsSearcher) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    const auto searcher = atlas::system::SpatialHashingSearcher<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .make_host_shared();

    ASSERT_NE(searcher, nullptr);
    EXPECT_FLOAT_EQ(searcher->cell_size(), 0.5f);
}

TEST(SpatialHashingSearcher, LinearKeyMatchesFlattenedIndexing) {
    const auto key = atlas::system::SpatialHashingSearcher<T>::linear_key(1, 2, 1, IVec3(4, 5, 6));

    EXPECT_EQ(key, static_cast<std::uint32_t>(1 + 2 * 4 + 1 * 4 * 5));
}

TEST(SpatialHashingSearcher, ResetIsSafeWithEmptyFluid) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    auto searcher = atlas::system::SpatialHashingSearcher<T>::builder()
                        .with_universe(universe)
                        .with_fluid(fluid)
                        .build();

    EXPECT_NO_THROW(searcher.reset());
    EXPECT_NE(searcher.cell_start(), nullptr);
    EXPECT_NE(searcher.cell_end(), nullptr);
}
