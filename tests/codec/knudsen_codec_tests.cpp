#include "../utilities/test_utils.h"

#include <atlas/codec/knudsen_codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <testkit/testkit.h>

namespace {

using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::KnudsenCodec;
using atlas::SpatialHashingSearcher;
using atlas::SpatialHashingSearcherHostPtr;
using atlas::tol;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::UniverseKnudsenNumberState;
using atlas::UniverseNumberParticleState;
using atlas::UniverseTemperatureState;

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
    // Create a small fluid with a known statistical weight.
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .with_statistical_weight(2.0f)
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

TEST(KnudsenCodec, BuilderConstructsCodecAndInitializesKnudsenState) {
    // Arrange: create all required codec dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build a Knudsen codec with a valid characteristic length.
    auto codec = KnudsenCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .build();

    // Assert: per-cell solver storage and Knudsen state are initialized.
    EXPECT_EQ(codec.allocated_solver().size(), static_cast<std::size_t>(universe->number_of_cells()));
    ASSERT_TRUE(universe->has_state<UniverseKnudsenNumberState<float>>());
}

TEST(KnudsenCodec, BuilderRejectsMissingDependenciesAndInvalidLength) {
    // Arrange: create valid dependencies used as controls.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Assert: missing universe dependency is rejected.
    EXPECT_THROW(
        KnudsenCodec<float>::builder()
            .with_fluid(fluid)
            .with_searcher(searcher)
            .with_characteristic_length(1.0f)
            .build(),
        std::invalid_argument);

    // Assert: non-positive characteristic length is rejected.
    EXPECT_THROW(
        KnudsenCodec<float>::builder()
            .with_domain(universe)
            .with_fluid(fluid)
            .with_searcher(searcher)
            .with_characteristic_length(0.0f)
            .build(),
        std::invalid_argument);
}

TEST(KnudsenCodec, UpdateComputesKnudsenNumberAndAllocatesSolverBuckets) {
    // Arrange: create dependencies and the universe states required by encoding.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));

    // Use equal temperatures and decreasing particle counts to produce increasing Knudsen numbers.
    temperature.data()[0] = 300.0f;
    temperature.data()[1] = 300.0f;
    temperature.data()[2] = 300.0f;

    number_particle.data()[0] = 1.0e26f;
    number_particle.data()[1] = 1.0e24f;
    number_particle.data()[2] = 1.0e22f;

    auto codec = KnudsenCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .build();

    // Act: compute Knudsen numbers and decode them into solver buckets.
    codec.update();

    const auto* knudsen_number = universe->state<UniverseKnudsenNumberState<float>>();

    // Assert: output state and solver allocation buffers are available.
    ASSERT_NE(knudsen_number, nullptr);
    ASSERT_GE(knudsen_number->data().size(), 3u);
    ASSERT_GE(codec.allocated_solver().size(), 3u);

    // Assert: Knudsen values are non-negative and solver buckets follow the expected ordering.
    EXPECT_GE(knudsen_number->data()[0], static_cast<float>(0));
    EXPECT_GE(knudsen_number->data()[1], static_cast<float>(0));
    EXPECT_GE(knudsen_number->data()[2], static_cast<float>(0));
    EXPECT_LE(codec.allocated_solver()[0], codec.allocated_solver()[1]);
    EXPECT_LE(codec.allocated_solver()[1], codec.allocated_solver()[2]);
}

TEST(KnudsenCodec, FixedRegionSkipsEncodingAndDecodesFixedSolver) {
    // Arrange: create dependencies and required universe states.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));

    temperature.data()[0]     = 300.0f;
    temperature.data()[1]     = 300.0f;
    number_particle.data()[0] = 1.0e26f;
    number_particle.data()[1] = 1.0e26f;

    // Mark cell 1 as fixed and assign a fixed solver id.
    DeviceBuffer<int> fixed_solver(static_cast<std::size_t>(universe->number_of_cells()), 0);
    DeviceBuffer<int> fixed_region(static_cast<std::size_t>(universe->number_of_cells()), 0);
    fixed_solver[1] = 7;
    fixed_region[1] = 1;

    auto codec = KnudsenCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(1.0f)
                     .with_fixed_solver(fixed_solver)
                     .with_fixed_region(fixed_region)
                     .build();

    // Seed cell 1 with a sentinel value that should not be overwritten.
    auto* knudsen_number = universe->state<UniverseKnudsenNumberState<float>>();
    ASSERT_NE(knudsen_number, nullptr);
    knudsen_number->data()[1] = 123.0f;

    // Act: update the codec.
    codec.update();

    // Assert: fixed cell keeps its Knudsen value and receives the fixed solver id.
    EXPECT_NEAR(knudsen_number->data()[1], 123.0f, 0.0f);
    EXPECT_EQ(codec.allocated_solver()[1], 7);
    EXPECT_NE(codec.allocated_solver()[0], 7);
}

TEST(KnudsenCodec, EncodeUsesFluidStatisticalWeightForNumberDensity) {
    // Arrange: create a fluid with a known statistical weight.
    const auto universe = make_universe();
    const auto fluid    = Fluid<float>::builder()
                           .with_buffer_size(8)
                           .with_statistical_weight(4.0f)
                           .make_host_shared();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));

    temperature.data()[0]     = 300.0f;
    number_particle.data()[0] = 2.0f;

    auto codec = KnudsenCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(2.0f)
                     .build();

    // Act: compute the Knudsen number from particle count and statistical weight.
    codec.update();

    const auto* knudsen_number = universe->state<UniverseKnudsenNumberState<float>>();

    ASSERT_NE(knudsen_number, nullptr);

    // Assert: number density uses physical particle count, not simulation particle count alone.
    const float expected_number_density = number_particle.data()[0] * fluid->statistical_weight() / universe->cell_volume();
    const float expected_knudsen_number = 1.0f
        / (static_cast<float>(atlas::SQRT_TWO) * expected_number_density)
        / 2.0f;

    EXPECT_NEAR(knudsen_number->data()[0], expected_knudsen_number, tol);
}

TEST(KnudsenCodec, EncodeUsesRepresentativeCollisionCrossSectionalArea) {
    const auto universe = make_universe();
    const auto fluid    = Fluid<float>::builder()
                           .with_buffer_size(8)
                           .with_statistical_weight(4.0f)
                           .make_host_shared();
    const auto searcher = make_searcher(universe, fluid);

    auto& temperature = universe->emplace_state<UniverseTemperatureState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));
    auto& number_particle = universe->emplace_state<UniverseNumberParticleState<float>>(
        static_cast<std::size_t>(universe->number_of_cells()));

    temperature.data()[0]     = 300.0f;
    number_particle.data()[0] = 2.0f;

    auto codec = KnudsenCodec<float>::builder()
                     .with_domain(universe)
                     .with_fluid(fluid)
                     .with_searcher(searcher)
                     .with_characteristic_length(2.0f)
                     .with_representative_collision_cross_sectional_area(4.0f)
                     .build();

    codec.update();

    const auto* knudsen_number = universe->state<UniverseKnudsenNumberState<float>>();
    ASSERT_NE(knudsen_number, nullptr);

    const float expected_number_density = number_particle.data()[0] * fluid->statistical_weight() / universe->cell_volume();
    const float expected_knudsen_number = 1.0f
        / (static_cast<float>(atlas::SQRT_TWO) * expected_number_density * 4.0f)
        / 2.0f;

    EXPECT_NEAR(knudsen_number->data()[0], expected_knudsen_number, tol);
}
