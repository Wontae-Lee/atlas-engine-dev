#include "../utilities/test_utils.h"

#include <atlas/codec/deep_learning_codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

namespace {

using atlas::DeepLearningCodec;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::SpatialHashingSearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;

UniverseHostPtr<float>
make_universe() {
    // Build a small 2 x 2 x 2 universe.
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

FluidHostPtr<float>
make_fluid() {
    // Allocate a small fluid buffer for codec construction tests.
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

SpatialHashingSearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    // Create the searcher dependency required by the codec.
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(DeepLearningCodec, BuilderConstructsCodecWithAllocatedSolverBuffer) {
    // Arrange: create all required codec dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the codec through the builder.
    auto codec = DeepLearningCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .build();

    // Assert: allocated solver storage is created per universe cell.
    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
}

TEST(DeepLearningCodec, BuilderAppliesFixedBuffers) {
    // Arrange: create dependencies and fixed per-cell buffers.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DeviceBuffer<int> fixed_solver(static_cast<std::size_t>(universe->number_of_cells()), 3);
    DeviceBuffer<int> fixed_region(static_cast<std::size_t>(universe->number_of_cells()), 1);

    // Act: build the codec with fixed solver and region buffers.
    auto codec = DeepLearningCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_fixed_solver(fixed_solver)
                     .with_fixed_region(fixed_region)
                     .build();

    // Assert: fixed buffers are copied into the codec.
    EXPECT_EQ(codec.fixed_solver()[0], 3);
    EXPECT_EQ(codec.fixed_region()[0], 1);
}

TEST(DeepLearningCodec, BuilderRejectsMissingDependencies) {
    // Arrange: create valid dependencies used as controls.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Assert: missing universe dependency is rejected.
    EXPECT_THROW(
        DeepLearningCodec<float>::builder()
            .with_fluid(fluid)
            .with_searcher(searcher)
            .build(),
        std::invalid_argument);

    // Assert: missing fluid dependency is rejected.
    EXPECT_THROW(
        DeepLearningCodec<float>::builder()
            .with_domain(universe)
            .with_searcher(searcher)
            .build(),
        std::invalid_argument);

    // Assert: missing searcher dependency is rejected.
    EXPECT_THROW(
        DeepLearningCodec<float>::builder()
            .with_domain(universe)
            .with_fluid(fluid)
            .build(),
        std::invalid_argument);
}

TEST(DeepLearningCodec, MakeHostSharedBuildsCodec) {
    // Arrange: create all required codec dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the codec as a host shared pointer.
    const auto codec = DeepLearningCodec<float>::builder()
                           .with_domain(universe)
                           .with_fluid(fluid)
                           .with_searcher(searcher)
                           .make_host_shared();

    // Assert: shared construction succeeds and the codec is usable.
    ASSERT_NE(codec, nullptr);
    EXPECT_EQ(codec->allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_NO_THROW(codec->update());
}