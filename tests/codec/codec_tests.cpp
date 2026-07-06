#include <atlas/codec/codec.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

using atlas::Codec;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3;

class DummyCodec final : public Codec {
public:
    DummyCodec(UniverseHostPtr universe,
               FluidHostPtr fluid,
               SearcherHostPtr searcher)
        : Codec(std::move(universe), std::move(fluid), std::move(searcher)) { }

    void
    encode() override {
        ++encode_calls;
    }

    void
    decode() override {
        ++decode_calls;
    }

    int encode_calls = 0;
    int decode_calls = 0;
};

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

TEST(Codec, ConstructorAllocatesSolverBufferForUniverseCells) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->cell_count()));
    EXPECT_EQ(codec.fixed_solver().size(), static_cast<std::size_t>(universe->cell_count()));
    EXPECT_EQ(codec.fixed_region().size(), static_cast<std::size_t>(universe->cell_count()));
}

TEST(Codec, UpdateInvokesEncodeAndDecode) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    codec.update();

    EXPECT_EQ(codec.encode_calls, 1);
    EXPECT_EQ(codec.decode_calls, 1);
}

TEST(Codec, ConstructorRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    EXPECT_THROW(DummyCodec(nullptr, fluid, searcher), std::invalid_argument);
    EXPECT_THROW(DummyCodec(universe, nullptr, searcher), std::invalid_argument);
    EXPECT_THROW(DummyCodec(universe, fluid, nullptr), std::invalid_argument);
}

TEST(Codec, FixedBuffersCanBeReplacedAndValidateCellCount) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    DeviceBuffer<int> fixed_solver(static_cast<std::size_t>(universe->cell_count()), 2);
    DeviceBuffer<int> fixed_region(static_cast<std::size_t>(universe->cell_count()), 1);

    codec.set_fixed_solver(fixed_solver);
    codec.set_fixed_region(fixed_region);

    EXPECT_EQ(codec.fixed_solver()[0], 2);
    EXPECT_EQ(codec.fixed_region()[0], 1);

    DeviceBuffer<int> wrong_size(1, 0);
    EXPECT_THROW(codec.set_fixed_solver(wrong_size), std::invalid_argument);
    EXPECT_THROW(codec.set_fixed_region(wrong_size), std::invalid_argument);
}
