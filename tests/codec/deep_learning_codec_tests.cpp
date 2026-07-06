#include <atlas/codec/deep_learning_codec.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::DeepLearningCodec;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3;

UniverseHostPtr
make_universe() {
    return Universe::builder()
        .with_lower_corner(Vector3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
        .with_cell_size(0.5f)
        .make_host_shared();
}

FluidHostPtr
make_fluid() {
    return Fluid::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

SearcherHostPtr
make_searcher(const UniverseHostPtr& universe, const FluidHostPtr& fluid) {
    return SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

}

TEST(DeepLearningCodec, BuilderConstructsCodecWithAllocatedSolverBuffer) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto codec = DeepLearningCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .build();

    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->cell_count()));
}

TEST(DeepLearningCodec, BuilderAppliesFixedBuffers) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DeviceBuffer<int> fixed_solver(static_cast<std::size_t>(universe->cell_count()), 3);
    DeviceBuffer<int> fixed_region(static_cast<std::size_t>(universe->cell_count()), 1);

    auto codec = DeepLearningCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_fixed_solver(fixed_solver)
                     .with_fixed_region(fixed_region)
                     .build();

    EXPECT_EQ(codec.fixed_solver()[0], 3);
    EXPECT_EQ(codec.fixed_region()[0], 1);
}

TEST(DeepLearningCodec, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    EXPECT_THROW(
        static_cast<void>(DeepLearningCodec::builder()
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .build()),
        std::invalid_argument);

    EXPECT_THROW(
        static_cast<void>(DeepLearningCodec::builder()
                              .with_domain(universe)
                              .with_searcher(searcher)
                              .build()),
        std::invalid_argument);

    EXPECT_THROW(
        static_cast<void>(DeepLearningCodec::builder()
                              .with_domain(universe)
                              .with_fluid(fluid)
                              .build()),
        std::invalid_argument);
}

TEST(DeepLearningCodec, MakeHostSharedBuildsCodec) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto codec = DeepLearningCodec::builder()
                           .with_domain(universe)
                           .with_fluid(fluid)
                           .with_searcher(searcher)
                           .make_host_shared();

    ASSERT_NE(codec, nullptr);
    EXPECT_EQ(codec->allocated_solver().size(), static_cast<std::size_t>(universe->cell_count()));
    EXPECT_NO_THROW(codec->update());
}
