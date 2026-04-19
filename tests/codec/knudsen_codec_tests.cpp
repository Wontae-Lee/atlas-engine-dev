#include "../utilities/tests_utils.h"

#include <atlas/codec/knudsen_codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <testkit/testkit.h>

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
        .with_statistical_weight(2.0f)
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

TEST(KnudsenCodec, BuilderConstructsCodecAndInitializesKnudsenState) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto codec = atlas::system::KnudsenCodec<T>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .build();

    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseKnudsenNumberState<T>>());
}

TEST(KnudsenCodec, BuilderRejectsMissingDependenciesAndInvalidLength) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    EXPECT_THROW(
        atlas::system::KnudsenCodec<T>::builder()
            .with_fluid(fluid)
            .with_searcher(searcher)
            .with_characteristic_length(1.0f)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::system::KnudsenCodec<T>::builder()
            .with_domain(universe)
            .with_fluid(fluid)
            .with_searcher(searcher)
            .with_characteristic_length(0.0f)
            .build(),
        std::invalid_argument);
}

TEST(KnudsenCodec, UpdateComputesKnudsenNumberAndAllocatesSolverBuckets) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<atlas::universe::UniverseTemperatureState<T>>(
        static_cast<std::size_t>(universe->number_of_cells()));
    auto& number_particle = universe->emplace_state<atlas::universe::UniverseNumberParticleState<T>>(
        static_cast<std::size_t>(universe->number_of_cells()));

    temperature.data()[0] = 300.0f;
    temperature.data()[1] = 300.0f;
    temperature.data()[2] = 300.0f;

    number_particle.data()[0] = 1.0e26f;
    number_particle.data()[1] = 1.0e24f;
    number_particle.data()[2] = 1.0e22f;

    auto codec = atlas::system::KnudsenCodec<T>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .build();

    codec.update();

    const auto* knudsen_number
        = universe->state<atlas::universe::UniverseKnudsenNumberState<T>>();

    ASSERT_NE(knudsen_number, nullptr);
    ASSERT_GE(knudsen_number->data().size(), 3u);
    ASSERT_GE(codec.allocated_solver().size(), 3u);

    EXPECT_GE(knudsen_number->data()[0], static_cast<T>(0));
    EXPECT_GE(knudsen_number->data()[1], static_cast<T>(0));
    EXPECT_GE(knudsen_number->data()[2], static_cast<T>(0));
    EXPECT_LE(codec.allocated_solver()[0], codec.allocated_solver()[1]);
    EXPECT_LE(codec.allocated_solver()[1], codec.allocated_solver()[2]);
}

TEST(KnudsenCodec, EncodeUsesFluidStatisticalWeightForNumberDensity) {
    const auto universe = make_universe();
    const auto fluid = atlas::fluid::Fluid<T>::builder()
                           .with_buffer_size(8)
                           .with_statistical_weight(4.0f)
                           .make_host_shared();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<atlas::universe::UniverseTemperatureState<T>>(
        static_cast<std::size_t>(universe->number_of_cells()));
    auto& number_particle = universe->emplace_state<atlas::universe::UniverseNumberParticleState<T>>(
        static_cast<std::size_t>(universe->number_of_cells()));

    temperature.data()[0] = 300.0f;
    number_particle.data()[0] = 2.0f;

    auto codec = atlas::system::KnudsenCodec<T>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(2.0f)
                     .build();

    codec.update();

    const auto* knudsen_number
        = universe->state<atlas::universe::UniverseKnudsenNumberState<T>>();

    ASSERT_NE(knudsen_number, nullptr);

    const T expected_number_density = number_particle.data()[0] * fluid->statistical_weight() / universe->cell_volume();
    const T expected_knudsen_number
        = static_cast<T>(atlas::boltzmann_constant) * temperature.data()[0]
        / expected_number_density
        / 2.0f;

    EXPECT_NEAR(knudsen_number->data()[0], expected_knudsen_number, static_cast<T>(1e-12));
}
