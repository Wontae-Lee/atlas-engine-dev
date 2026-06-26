#include "../utilities/test_utils.h"

#include <atlas/codec/codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

namespace {

using atlas::Codec;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::SearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;

/**
 * @brief Minimal codec implementation used to verify base Codec behavior.
 */
class DummyCodec final : public Codec<float> {
public:
    DummyCodec(UniverseHostPtr<float> universe,
               FluidHostPtr<float> fluid,
               SearcherHostPtr<float> searcher)
        : Codec<float>(std::move(universe), std::move(fluid), std::move(searcher)) { }

    void
    encode() override {
        // Count encode invocations made through Codec::update().
        ++encode_calls;
    }

    void
    decode() override {
        // Count decode invocations made through Codec::update().
        ++decode_calls;
    }

    int encode_calls = 0;
    int decode_calls = 0;
};

UniverseHostPtr<float>
make_universe() {
    // Build a small 2 x 2 x 2 cell universe.
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

FluidHostPtr<float>
make_fluid() {
    // Allocate a small fluid buffer sufficient for codec construction tests.
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

SearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    // Create the searcher dependency required by Codec.
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(Codec, ConstructorAllocatesSolverBufferForUniverseCells) {
    // Arrange: create the required codec dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct the codec.
    DummyCodec codec(universe, fluid, searcher);

    // Assert: per-cell codec buffers match the universe cell count.
    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_EQ(codec.fixed_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_EQ(codec.fixed_region().size(), static_cast<std::size_t>(universe->number_of_cells()));
}

TEST(Codec, UpdateInvokesEncodeAndDecode) {
    // Arrange: create a codec that counts virtual method calls.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    // Act: run the codec update pipeline.
    codec.update();

    // Assert: update invokes encode() and decode() exactly once.
    EXPECT_EQ(codec.encode_calls, 1);
    EXPECT_EQ(codec.decode_calls, 1);
}

TEST(Codec, ConstructorRejectsMissingDependencies) {
    // Arrange: create valid dependencies used as controls.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Assert: every required dependency must be non-null.
    EXPECT_THROW(DummyCodec(nullptr, fluid, searcher), std::invalid_argument);
    EXPECT_THROW(DummyCodec(universe, nullptr, searcher), std::invalid_argument);
    EXPECT_THROW(DummyCodec(universe, fluid, nullptr), std::invalid_argument);
}

TEST(Codec, FixedBuffersCanBeReplacedAndValidateCellCount) {
    // Arrange: create a codec with valid per-cell buffer sizes.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    DeviceBuffer<int> fixed_solver(static_cast<std::size_t>(universe->number_of_cells()), 2);
    DeviceBuffer<int> fixed_region(static_cast<std::size_t>(universe->number_of_cells()), 1);

    // Act: replace fixed solver and fixed region buffers.
    codec.set_fixed_solver(fixed_solver);
    codec.set_fixed_region(fixed_region);

    // Assert: replacement buffers are stored.
    EXPECT_EQ(codec.fixed_solver()[0], 2);
    EXPECT_EQ(codec.fixed_region()[0], 1);

    // Assert: buffers with the wrong cell count are rejected.
    DeviceBuffer<int> wrong_size(1, 0);
    EXPECT_THROW(codec.set_fixed_solver(wrong_size), std::invalid_argument);
    EXPECT_THROW(codec.set_fixed_region(wrong_size), std::invalid_argument);
}