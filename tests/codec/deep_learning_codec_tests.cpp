#include "../utilities/tests_utils.h"

#include <atlas/codec/deep_learning_codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

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

atlas::SpatialHashingSearcherHostPtr<T>
make_searcher(const atlas::UniverseHostPtr<T>& universe,
              const atlas::FluidHostPtr<T>& fluid) {
    return atlas::system::SpatialHashingSearcher<T>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(DeepLearningCodec, BuilderConstructsCodecWithAllocatedSolverBuffer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto codec = atlas::system::DeepLearningCodec<T>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .build();

    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
}

TEST(DeepLearningCodec, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    EXPECT_THROW(
        atlas::system::DeepLearningCodec<T>::builder()
            .with_fluid(fluid)
            .with_searcher(searcher)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::system::DeepLearningCodec<T>::builder()
            .with_domain(universe)
            .with_searcher(searcher)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::system::DeepLearningCodec<T>::builder()
            .with_domain(universe)
            .with_fluid(fluid)
            .build(),
        std::invalid_argument);
}

TEST(DeepLearningCodec, MakeHostSharedBuildsCodec) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto codec = atlas::system::DeepLearningCodec<T>::builder()
                           .with_domain(universe)
                           .with_fluid(fluid)
                           .with_searcher(searcher)
                           .make_host_shared();

    ASSERT_NE(codec, nullptr);
    EXPECT_EQ(codec->allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_NO_THROW(codec->update());
}
