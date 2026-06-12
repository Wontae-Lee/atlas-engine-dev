#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/measure/boltzman_measurer.h>

#include <testkit/testkit.h>

namespace {

using atlas::BoltzmanMeasurer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::MeasureModeType;
using atlas::SpatialHashingSearcher;
using atlas::SpatialHashingSearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::FluidTemperatureState;
using atlas::UniverseBulkVelocityState;
using atlas::UniverseTemperatureState;
using atlas::UniverseThermalEnergyState;

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

FluidHostPtr<float>
make_fluid() {
    return Fluid<float>::builder()
        .with_buffer_size(4)
        .make_host_shared();
}

SpatialHashingSearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(BoltzmanMeasurer, ConstructorCreatesRequiredUniverseAndFluidStates) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    ASSERT_FALSE(universe->has_state<UniverseTemperatureState<float>>());
    ASSERT_FALSE(universe->has_state<UniverseBulkVelocityState<float>>());
    ASSERT_FALSE(universe->has_state<UniverseThermalEnergyState<float>>());
    ASSERT_FALSE(fluid->has_state<FluidTemperatureState<float>>());

    const BoltzmanMeasurer<float> measurer(universe, fluid, searcher, MeasureModeType::All);

    EXPECT_EQ(measurer.measure_mode(), MeasureModeType::All);
    ASSERT_TRUE(universe->has_state<UniverseTemperatureState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseBulkVelocityState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseThermalEnergyState<float>>());
    ASSERT_TRUE(fluid->has_state<FluidTemperatureState<float>>());
    EXPECT_EQ(universe->state<UniverseTemperatureState<float>>()->size(), 8u);
    EXPECT_EQ(universe->state<UniverseBulkVelocityState<float>>()->size(), 8u);
    EXPECT_EQ(universe->state<UniverseThermalEnergyState<float>>()->size(), 8u);
    EXPECT_EQ(fluid->state<FluidTemperatureState<float>>()->size(), 4u);
}

TEST(BoltzmanMeasurer, BuilderConstructsUsableMeasurer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto measurer = BoltzmanMeasurer<float>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(MeasureModeType::Field)
                              .build();

    EXPECT_EQ(measurer.measure_mode(), MeasureModeType::Field);
}

TEST(BoltzmanMeasurer, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(
        BoltzmanMeasurer<float>::builder()
            .with_universe(universe)
            .with_fluid(fluid)
            .build(),
        std::runtime_error);
}

TEST(BoltzmanMeasurer, MakeHostSharedBuildsMeasurer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto measurer = BoltzmanMeasurer<float>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .make_host_shared();

    ASSERT_NE(measurer, nullptr);
    EXPECT_EQ(measurer->measure_mode(), MeasureModeType::Field);
}

TEST(BoltzmanMeasurer, MeasureIsSafeNoOpWithoutDependencies) {
    BoltzmanMeasurer<float> measurer;

    EXPECT_NO_THROW(measurer.measure());
}
