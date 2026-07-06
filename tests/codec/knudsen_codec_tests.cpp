#include <atlas/codec/knudsen_codec.h>

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/math/math.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace {

using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::KnudsenCodec;
using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::UniverseKnudsenNumberState;
using atlas::UniverseNumberParticleState;
using atlas::UniverseTemperatureState;
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
        .with_statistical_weight(2.0f)
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

TEST(KnudsenCodec, BuilderConstructsCodecAndInitializesKnudsenState) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto codec = KnudsenCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .build();

    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->cell_count()));
    ASSERT_TRUE(universe->has_state<UniverseKnudsenNumberState>());
}

TEST(KnudsenCodec, BuilderRejectsMissingDependenciesAndInvalidLength) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    EXPECT_THROW(
        static_cast<void>(KnudsenCodec::builder()
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_characteristic_length(1.0f)
                              .build()),
        std::invalid_argument);

    EXPECT_THROW(
        static_cast<void>(KnudsenCodec::builder()
                              .with_domain(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_characteristic_length(0.0f)
                              .build()),
        std::invalid_argument);
}

TEST(KnudsenCodec, UpdateComputesKnudsenNumberAndAllocatesSolverBuckets) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState>(
        static_cast<std::size_t>(universe->cell_count()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState>(
        static_cast<std::size_t>(universe->cell_count()));

    temperature.data()[0] = 300.0f;
    temperature.data()[1] = 300.0f;
    temperature.data()[2] = 300.0f;

    number_particle.data()[0] = 1.0e26f;
    number_particle.data()[1] = 1.0e24f;
    number_particle.data()[2] = 1.0e22f;

    auto codec = KnudsenCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .build();

    codec.update();

    const auto* knudsen_number = universe->state<UniverseKnudsenNumberState>();

    ASSERT_NE(knudsen_number, nullptr);
    ASSERT_GE(knudsen_number->data().size(), 3u);
    ASSERT_GE(codec.allocated_solver().size(), 3u);

    EXPECT_GE(knudsen_number->data()[0], 0.0f);
    EXPECT_GE(knudsen_number->data()[1], 0.0f);
    EXPECT_GE(knudsen_number->data()[2], 0.0f);
    EXPECT_LE(codec.allocated_solver()[0], codec.allocated_solver()[1]);
    EXPECT_LE(codec.allocated_solver()[1], codec.allocated_solver()[2]);
}

TEST(KnudsenCodec, FixedRegionSkipsEncodingAndDecodesFixedSolver) {
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState>(
        static_cast<std::size_t>(universe->cell_count()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState>(
        static_cast<std::size_t>(universe->cell_count()));

    temperature.data()[0]     = 300.0f;
    temperature.data()[1]     = 300.0f;
    number_particle.data()[0] = 1.0e26f;
    number_particle.data()[1] = 1.0e26f;

    DeviceBuffer<int> fixed_solver(static_cast<std::size_t>(universe->cell_count()), 0);
    DeviceBuffer<int> fixed_region(static_cast<std::size_t>(universe->cell_count()), 0);
    fixed_solver[1] = 7;
    fixed_region[1] = 1;

    auto codec = KnudsenCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .with_fixed_solver(fixed_solver)
                     .with_fixed_region(fixed_region)
                     .build();

    auto* knudsen_number = universe->state<UniverseKnudsenNumberState>();
    ASSERT_NE(knudsen_number, nullptr);
    knudsen_number->data()[1] = 123.0f;

    codec.update();

    EXPECT_NEAR(knudsen_number->data()[1], 123.0f, 0.0f);
    EXPECT_EQ(codec.allocated_solver()[1], 7);
    EXPECT_NE(codec.allocated_solver()[0], 7);
}

TEST(KnudsenCodec, EncodeUsesFluidStatisticalWeightForNumberDensity) {
    const auto universe = make_universe();
    const auto fluid    = Fluid::builder()
                           .with_buffer_size(8)
                           .with_statistical_weight(4.0f)
                           .make_host_shared();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState>(
        static_cast<std::size_t>(universe->cell_count()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState>(
        static_cast<std::size_t>(universe->cell_count()));

    temperature.data()[0]     = 300.0f;
    number_particle.data()[0] = 2.0f;

    auto codec = KnudsenCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(2.0f)
                     .build();

    codec.update();

    const auto* knudsen_number = universe->state<UniverseKnudsenNumberState>();

    ASSERT_NE(knudsen_number, nullptr);

    const float expected_number_density = number_particle.data()[0] * fluid->statistical_weight() / universe->cell_volume();
    const float expected_knudsen_number = 1.0f
        / (atlas::SQRT_TWO * expected_number_density)
        / 2.0f;

    EXPECT_NEAR(knudsen_number->data()[0], expected_knudsen_number, atlas::tol);
}

TEST(KnudsenCodec, EncodeUsesRepresentativeCollisionCrossSectionalArea) {
    const auto universe = make_universe();
    const auto fluid    = Fluid::builder()
                           .with_buffer_size(8)
                           .with_statistical_weight(4.0f)
                           .make_host_shared();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState>(
        static_cast<std::size_t>(universe->cell_count()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState>(
        static_cast<std::size_t>(universe->cell_count()));

    temperature.data()[0]     = 300.0f;
    number_particle.data()[0] = 2.0f;

    auto codec = KnudsenCodec::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(2.0f)
                     .with_representative_collision_cross_sectional_area(4.0f)
                     .build();

    codec.update();

    const auto* knudsen_number = universe->state<UniverseKnudsenNumberState>();
    ASSERT_NE(knudsen_number, nullptr);

    const float expected_number_density = number_particle.data()[0] * fluid->statistical_weight() / universe->cell_volume();
    const float expected_knudsen_number = 1.0f
        / (atlas::SQRT_TWO * expected_number_density * 4.0f)
        / 2.0f;

    EXPECT_NEAR(knudsen_number->data()[0], expected_knudsen_number, atlas::tol);
}
