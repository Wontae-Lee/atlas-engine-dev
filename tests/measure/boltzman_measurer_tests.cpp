#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/measure/boltzman_measurer.h>

#include <gtest/gtest.h>

namespace {

using T = float;

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(atlas::Vector3<T>(0, 0, 0))
        .with_upper_corner(atlas::Vector3<T>(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(4)
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

TEST(BoltzmanMeasurer, ConstructorCreatesRequiredUniverseAndFluidStates) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    ASSERT_FALSE(universe->has_state<atlas::universe::UniverseTemperatureState<T>>());
    ASSERT_FALSE(universe->has_state<atlas::universe::UniverseBulkVelocityState<T>>());
    ASSERT_FALSE(universe->has_state<atlas::universe::UniverseMomentumWeightState<T>>());
    ASSERT_FALSE(universe->has_state<atlas::universe::UniverseThermalEnergyState<T>>());
    ASSERT_FALSE(fluid->has_state<atlas::fluid::FluidTemperatureState<T>>());

    const atlas::system::BoltzmanMeasurer<T> measurer(universe, fluid, searcher, atlas::MeasureModeType::All);

    EXPECT_EQ(measurer.measure_mode(), atlas::MeasureModeType::All);
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseTemperatureState<T>>());
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseBulkVelocityState<T>>());
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseMomentumWeightState<T>>());
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseThermalEnergyState<T>>());
    ASSERT_TRUE(fluid->has_state<atlas::fluid::FluidTemperatureState<T>>());
    EXPECT_EQ(universe->state<atlas::universe::UniverseTemperatureState<T>>()->size(), 8u);
    EXPECT_EQ(universe->state<atlas::universe::UniverseBulkVelocityState<T>>()->size(), 8u);
    EXPECT_EQ(universe->state<atlas::universe::UniverseMomentumWeightState<T>>()->size(), 8u);
    EXPECT_EQ(universe->state<atlas::universe::UniverseThermalEnergyState<T>>()->size(), 8u);
    EXPECT_EQ(fluid->state<atlas::fluid::FluidTemperatureState<T>>()->size(), 4u);
}

TEST(BoltzmanMeasurer, BuilderConstructsUsableMeasurer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto measurer = atlas::system::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .with_measure_mode(atlas::MeasureModeType::Field)
                              .build();

    EXPECT_EQ(measurer.measure_mode(), atlas::MeasureModeType::Field);
}

TEST(BoltzmanMeasurer, BuilderRejectsMissingDependencies) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::system::BoltzmanMeasurer<T>::builder()
            .with_universe(universe)
            .with_fluid(fluid)
            .build(),
        std::runtime_error);
}

TEST(BoltzmanMeasurer, MakeHostSharedBuildsMeasurer) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto measurer = atlas::system::BoltzmanMeasurer<T>::builder()
                              .with_universe(universe)
                              .with_fluid(fluid)
                              .with_searcher(searcher)
                              .make_host_shared();

    ASSERT_NE(measurer, nullptr);
    EXPECT_EQ(measurer->measure_mode(), atlas::MeasureModeType::Field);
}

TEST(BoltzmanMeasurer, MeasureIsSafeNoOpWithoutDependencies) {
    atlas::system::BoltzmanMeasurer<T> measurer;

    EXPECT_NO_THROW(measurer.measure());
}
