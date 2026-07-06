#include <atlas/measure/boltzmann_measurer.h>

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

using atlas::BoltzmannMeasurer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidTemperatureState;
using atlas::MeasureModeType;
using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::Universe;
using atlas::UniverseBulkVelocityState;
using atlas::UniverseHostPtr;
using atlas::UniverseTemperatureState;
using atlas::UniverseThermalEnergyState;
using atlas::Vector3;

UniverseHostPtr
make_universe() {
    return Universe::builder()
        .with_lower_corner(Vector3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f)
        .make_host_shared();
}

FluidHostPtr
make_fluid() {
    return Fluid::builder()
        .with_buffer_size(4)
        .make_host_shared();
}

SearcherHostPtr
make_searcher(const UniverseHostPtr& universe,
              const FluidHostPtr& fluid) {
    return SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

}

TEST(BoltzmannMeasurer, ConstructorCreatesRequiredUniverseAndFluidStates) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    ASSERT_FALSE(universe->has_state<UniverseTemperatureState>());
    ASSERT_FALSE(universe->has_state<UniverseBulkVelocityState>());
    ASSERT_FALSE(universe->has_state<UniverseThermalEnergyState>());
    ASSERT_FALSE(fluid->has_state<FluidTemperatureState>());

    const BoltzmannMeasurer measurer(universe, fluid, searcher, MeasureModeType::all);

    EXPECT_EQ(measurer.measure_mode(), MeasureModeType::all);
    ASSERT_TRUE(universe->has_state<UniverseTemperatureState>());
    ASSERT_TRUE(universe->has_state<UniverseBulkVelocityState>());
    ASSERT_TRUE(universe->has_state<UniverseThermalEnergyState>());
    ASSERT_TRUE(fluid->has_state<FluidTemperatureState>());
    EXPECT_EQ(universe->state<UniverseTemperatureState>()->size(), 8u);
    EXPECT_EQ(universe->state<UniverseBulkVelocityState>()->size(), 8u);
    EXPECT_EQ(universe->state<UniverseThermalEnergyState>()->size(), 8u);
    EXPECT_EQ(fluid->state<FluidTemperatureState>()->size(), 4u);
}

TEST(BoltzmannMeasurer, BuilderConstructsUsableMeasurer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto measurer = BoltzmannMeasurer::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(MeasureModeType::field)
                              .build();

    EXPECT_EQ(measurer.measure_mode(), MeasureModeType::field);
}

TEST(BoltzmannMeasurer, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(
        static_cast<void>(BoltzmannMeasurer::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .build()),
        std::runtime_error);
}

TEST(BoltzmannMeasurer, MakeHostSharedBuildsMeasurer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto measurer = BoltzmannMeasurer::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .make_host_shared();

    ASSERT_NE(measurer, nullptr);
    EXPECT_EQ(measurer->measure_mode(), MeasureModeType::field);
}

TEST(BoltzmannMeasurer, MeasureIsSafeNoOpWithoutDependencies) {
    BoltzmannMeasurer measurer;

    EXPECT_NO_THROW(measurer.measure());
}
