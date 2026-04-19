#include "../utilities/tests_utils.h"

#include <atlas/codec/codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

class DummyCodec final : public atlas::system::Codec<T> {
public:
    DummyCodec(atlas::UniverseHostPtr<T> universe,
               atlas::FluidHostPtr<T> fluid,
               atlas::SpatialHashingSearcherHostPtr<T> searcher)
        : atlas::system::Codec<T>(std::move(universe), std::move(fluid), std::move(searcher)) { }

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

TEST(Codec, ConstructorAllocatesSolverBufferForUniverseCells) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
}

TEST(Codec, UpdateInvokesEncodeAndDecode) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyCodec codec(universe, fluid, searcher);

    codec.update();

    EXPECT_EQ(codec.encode_calls, 1);
    EXPECT_EQ(codec.decode_calls, 1);
}

TEST(Codec, ConstructorRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    EXPECT_THROW(DummyCodec(nullptr, fluid, searcher), std::invalid_argument);
    EXPECT_THROW(DummyCodec(universe, nullptr, searcher), std::invalid_argument);
    EXPECT_THROW(DummyCodec(universe, fluid, nullptr), std::invalid_argument);
}
