#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>

#include <gtest/gtest.h>

namespace {

using T = float;

} // namespace

TEST(Fluid, DefaultConstructorStartsEmpty) {
    const atlas::fluid::Fluid<T> fluid;

    EXPECT_EQ(fluid.buffer_size(), 0u);
    EXPECT_EQ(fluid.particle_count(), 0u);
    EXPECT_FALSE(fluid.has_state<atlas::fluid::FluidPositionState<T>>());
    EXPECT_FALSE(fluid.has_state<atlas::fluid::FluidVelocityState<T>>());
    EXPECT_FALSE(fluid.has_state<atlas::fluid::FluidSpeciesState<T>>());
    EXPECT_FALSE(fluid.has_state<atlas::fluid::FluidActiveState<T>>());
}

TEST(Fluid, SizedConstructorRegistersDefaultStates) {
    const atlas::fluid::Fluid<T> fluid(8);

    ASSERT_TRUE(fluid.has_state<atlas::fluid::FluidPositionState<T>>());
    ASSERT_TRUE(fluid.has_state<atlas::fluid::FluidVelocityState<T>>());
    ASSERT_TRUE(fluid.has_state<atlas::fluid::FluidSpeciesState<T>>());
    ASSERT_TRUE(fluid.has_state<atlas::fluid::FluidActiveState<T>>());
    EXPECT_EQ(fluid.buffer_size(), 8u);
    EXPECT_EQ(fluid.particle_count(), 0u);
    EXPECT_EQ(fluid.state<atlas::fluid::FluidPositionState<T>>()->size(), 8u);
    EXPECT_EQ(fluid.state<atlas::fluid::FluidVelocityState<T>>()->size(), 8u);
    EXPECT_EQ(fluid.state<atlas::fluid::FluidSpeciesState<T>>()->size(), 8u);
    EXPECT_EQ(fluid.state<atlas::fluid::FluidActiveState<T>>()->size(), 8u);
}

TEST(Fluid, BuilderConstructsConfiguredFluid) {
    atlas::HostBuffer<atlas::system::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);
    properties[0].mass = 10.0f;
    properties[0].molecular_mass = 2.0f;

    const auto fluid = atlas::fluid::Fluid<T>::builder()
                           .with_buffer_size(16)
                           .with_statistical_weight(5.0f)
                           .with_properties(properties)
                           .with_generators(generators)
                           .build();

    EXPECT_EQ(fluid.buffer_size(), 16u);
    EXPECT_EQ(fluid.generators().size(), 1u);
    EXPECT_FLOAT_EQ(fluid.statistical_weight(), 5.0f);
    EXPECT_TRUE(fluid.has_state<atlas::fluid::FluidPositionState<T>>());
    EXPECT_TRUE(fluid.has_state<atlas::fluid::FluidVelocityState<T>>());
    EXPECT_TRUE(fluid.has_state<atlas::fluid::FluidSpeciesState<T>>());
    EXPECT_TRUE(fluid.has_state<atlas::fluid::FluidActiveState<T>>());
}

TEST(Fluid, BuilderRejectsMismatchedPropertiesAndGenerators) {
    atlas::HostBuffer<atlas::system::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(0);

    EXPECT_THROW(
        atlas::fluid::Fluid<T>::builder()
            .with_buffer_size(4)
            .with_properties(properties)
            .with_generators(generators)
            .build(),
        std::runtime_error);
}

TEST(Fluid, BuilderRejectsNonPositiveStatisticalWeight) {
    atlas::HostBuffer<atlas::system::MaterialProperties<T>> properties(0);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(0);

    EXPECT_THROW(
        atlas::fluid::Fluid<T>::builder()
            .with_buffer_size(4)
            .with_statistical_weight(0.0f)
            .with_properties(properties)
            .with_generators(generators)
            .build(),
        std::runtime_error);
}

TEST(Fluid, BuilderRejectsMassThatDoesNotMatchStatisticalWeight) {
    atlas::HostBuffer<atlas::system::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);
    properties[0].mass = 9.0f;
    properties[0].molecular_mass = 2.0f;

    EXPECT_THROW(
        atlas::fluid::Fluid<T>::builder()
            .with_buffer_size(4)
            .with_statistical_weight(5.0f)
            .with_properties(properties)
            .with_generators(generators)
            .build(),
        std::runtime_error);
}

TEST(Fluid, StateLifecycleSupportsInsertLookupReplaceAndRemove) {
    atlas::fluid::Fluid<T> fluid(4);

    EXPECT_FALSE(fluid.has_state<atlas::fluid::FluidTemperatureState<T>>());

    auto& state = fluid.emplace_state<atlas::fluid::FluidTemperatureState<T>>(4);
    EXPECT_TRUE(fluid.has_state<atlas::fluid::FluidTemperatureState<T>>());
    EXPECT_EQ(state.size(), 4u);
    ASSERT_NE(fluid.state<atlas::fluid::FluidTemperatureState<T>>(), nullptr);

    auto replacement = std::make_unique<atlas::fluid::FluidTemperatureState<T>>(6);
    fluid.set_state<atlas::fluid::FluidTemperatureState<T>>(std::move(replacement));

    ASSERT_NE(fluid.state<atlas::fluid::FluidTemperatureState<T>>(), nullptr);
    EXPECT_EQ(fluid.state<atlas::fluid::FluidTemperatureState<T>>()->size(), 6u);

    auto removed = fluid.remove_state<atlas::fluid::FluidTemperatureState<T>>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), 6u);
    EXPECT_FALSE(fluid.has_state<atlas::fluid::FluidTemperatureState<T>>());
    EXPECT_EQ(fluid.state<atlas::fluid::FluidTemperatureState<T>>(), nullptr);
}

TEST(Fluid, SetStateRejectsNullOwnershipTransfer) {
    atlas::fluid::Fluid<T> fluid(4);

    std::unique_ptr<atlas::fluid::FluidTemperatureState<T>> null_state;

    EXPECT_THROW(fluid.set_state<atlas::fluid::FluidTemperatureState<T>>(std::move(null_state)), std::invalid_argument);
}
