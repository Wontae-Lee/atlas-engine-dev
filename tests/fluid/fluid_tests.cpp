#include <atlas/fluid/fluid.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

TEST(Fluid, DefaultConstructorStartsEmpty) {
    const atlas::Fluid fluid;

    EXPECT_EQ(fluid.buffer_size(), 0u);
    EXPECT_EQ(fluid.particle_count(), 0u);
    EXPECT_FALSE(fluid.has_state<atlas::FluidPositionState>());
    EXPECT_FALSE(fluid.has_state<atlas::FluidVelocityState>());
    EXPECT_FALSE(fluid.has_state<atlas::FluidSpeciesState>());
    EXPECT_FALSE(fluid.has_state<atlas::FluidActiveState>());
}

TEST(Fluid, SizedConstructorRegistersDefaultStates) {
    const atlas::Fluid fluid(8);

    ASSERT_TRUE(fluid.has_state<atlas::FluidPositionState>());
    ASSERT_TRUE(fluid.has_state<atlas::FluidVelocityState>());
    ASSERT_TRUE(fluid.has_state<atlas::FluidSpeciesState>());
    ASSERT_TRUE(fluid.has_state<atlas::FluidActiveState>());
    EXPECT_EQ(fluid.buffer_size(), 8u);
    EXPECT_EQ(fluid.particle_count(), 0u);
    EXPECT_EQ(fluid.state<atlas::FluidPositionState>()->size(), 8u);
    EXPECT_EQ(fluid.state<atlas::FluidVelocityState>()->size(), 8u);
    EXPECT_EQ(fluid.state<atlas::FluidSpeciesState>()->size(), 8u);
    EXPECT_EQ(fluid.state<atlas::FluidActiveState>()->size(), 8u);
}

TEST(Fluid, BuilderConstructsConfiguredFluid) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);
    properties[0].mass           = 10.0f;
    properties[0].molecular_mass = 2.0f;

    const auto fluid = atlas::Fluid::builder()
                           .with_buffer_size(16)
                           .with_statistical_weight(5.0f)
                           .with_properties(properties)
                           .with_generators(generators)
                           .build();

    EXPECT_EQ(fluid.buffer_size(), 16u);
    EXPECT_EQ(fluid.generators().size(), 1u);
    EXPECT_FLOAT_EQ(fluid.statistical_weight(), 5.0f);
    EXPECT_TRUE(fluid.has_state<atlas::FluidPositionState>());
    EXPECT_TRUE(fluid.has_state<atlas::FluidVelocityState>());
    EXPECT_TRUE(fluid.has_state<atlas::FluidSpeciesState>());
    EXPECT_TRUE(fluid.has_state<atlas::FluidActiveState>());
}

TEST(Fluid, BuilderRejectsMismatchedPropertiesAndGenerators) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(0);

    EXPECT_THROW(
        static_cast<void>(atlas::Fluid::builder()
                              .with_buffer_size(4)
                              .with_properties(properties)
                              .with_generators(generators)
                              .build()),
        std::runtime_error);
}

TEST(Fluid, BuilderRejectsNonPositiveStatisticalWeight) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(0);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(0);

    EXPECT_THROW(
        static_cast<void>(atlas::Fluid::builder()
                              .with_buffer_size(4)
                              .with_statistical_weight(0.0f)
                              .with_properties(properties)
                              .with_generators(generators)
                              .build()),
        std::runtime_error);
}

TEST(Fluid, BuilderRejectsMassThatDoesNotMatchStatisticalWeight) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);
    properties[0].mass           = 9.0f;
    properties[0].molecular_mass = 2.0f;

    EXPECT_THROW(
        static_cast<void>(atlas::Fluid::builder()
                              .with_buffer_size(4)
                              .with_statistical_weight(5.0f)
                              .with_properties(properties)
                              .with_generators(generators)
                              .build()),
        std::runtime_error);
}

TEST(Fluid, SetParticleCountTracksActivePrefix) {
    atlas::Fluid fluid(4);

    EXPECT_NO_THROW(fluid.set_particle_count(3));
    EXPECT_EQ(fluid.particle_count(), 3u);

    EXPECT_THROW(fluid.set_particle_count(5), std::out_of_range);
}

TEST(Fluid, StateLifecycleSupportsInsertLookupReplaceAndRemove) {
    atlas::Fluid fluid(4);

    EXPECT_FALSE(fluid.has_state<atlas::FluidTemperatureState>());

    auto& state = fluid.emplace_state<atlas::FluidTemperatureState>(4);
    EXPECT_TRUE(fluid.has_state<atlas::FluidTemperatureState>());
    EXPECT_EQ(state.size(), 4u);
    ASSERT_NE(fluid.state<atlas::FluidTemperatureState>(), nullptr);

    auto replacement = std::make_unique<atlas::FluidTemperatureState>(6);
    fluid.set_state<atlas::FluidTemperatureState>(std::move(replacement));

    ASSERT_NE(fluid.state<atlas::FluidTemperatureState>(), nullptr);
    EXPECT_EQ(fluid.state<atlas::FluidTemperatureState>()->size(), 6u);

    auto removed = fluid.remove_state<atlas::FluidTemperatureState>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), 6u);
    EXPECT_FALSE(fluid.has_state<atlas::FluidTemperatureState>());
    EXPECT_EQ(fluid.state<atlas::FluidTemperatureState>(), nullptr);
}

TEST(Fluid, SetStateRejectsNullOwnershipTransfer) {
    atlas::Fluid fluid(4);

    std::unique_ptr<atlas::FluidTemperatureState> null_state;

    EXPECT_THROW(fluid.set_state<atlas::FluidTemperatureState>(std::move(null_state)), std::invalid_argument);
}

TEST(Fluid, SaveAndReloadBinarySnapshot) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);

    properties[0].type           = atlas::MaterialType::molecule;
    properties[0].mass           = 6.0f;
    properties[0].molecular_mass = 2.0f;
    properties[0].species_id     = 7;

    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(4)
                     .with_statistical_weight(3.0f)
                     .with_properties(properties)
                     .with_generators(generators)
                     .build();

    fluid.emplace_state<atlas::FluidTemperatureState>(4);
    fluid.set_particle_count(2);

    auto* position    = fluid.state<atlas::FluidPositionState>();
    auto* velocity    = fluid.state<atlas::FluidVelocityState>();
    auto* species     = fluid.state<atlas::FluidSpeciesState>();
    auto* active      = fluid.state<atlas::FluidActiveState>();
    auto* temperature = fluid.state<atlas::FluidTemperatureState>();

    ASSERT_NE(position, nullptr);
    ASSERT_NE(velocity, nullptr);
    ASSERT_NE(species, nullptr);
    ASSERT_NE(active, nullptr);
    ASSERT_NE(temperature, nullptr);

    position->data()[0]    = atlas::Vector3(1.0f, 2.0f, 3.0f);
    position->data()[1]    = atlas::Vector3(4.0f, 5.0f, 6.0f);
    velocity->data()[0]    = atlas::Vector3(0.1f, 0.2f, 0.3f);
    velocity->data()[1]    = atlas::Vector3(0.4f, 0.5f, 0.6f);
    species->data()[0]     = 0u;
    species->data()[1]     = 0u;
    active->data()[0]      = 1;
    active->data()[1]      = 1;
    temperature->data()[0] = 300.0f;
    temperature->data()[1] = 450.0f;

    const std::filesystem::path snapshot_path
        = std::filesystem::temp_directory_path() / "atlas_fluid_snapshot_test.bin";
    fluid.save(snapshot_path.string());

    const auto restored = atlas::Fluid::builder()
                              .with_binary(snapshot_path.string())
                              .build();

    EXPECT_EQ(restored.buffer_size(), 4u);
    EXPECT_EQ(restored.particle_count(), 2u);
    EXPECT_FLOAT_EQ(restored.statistical_weight(), 3.0f);
    ASSERT_EQ(restored.particle_properties().size(), 1u);
    const atlas::HostBuffer<atlas::MaterialProperties> restored_properties(
        restored.particle_properties().begin(),
        restored.particle_properties().end());
    EXPECT_EQ(restored_properties[0].species_id.value_or(-1), 7);

    const auto* restored_position    = restored.state<atlas::FluidPositionState>();
    const auto* restored_velocity    = restored.state<atlas::FluidVelocityState>();
    const auto* restored_active      = restored.state<atlas::FluidActiveState>();
    const auto* restored_temperature = restored.state<atlas::FluidTemperatureState>();

    ASSERT_NE(restored_position, nullptr);
    ASSERT_NE(restored_velocity, nullptr);
    ASSERT_NE(restored_active, nullptr);
    ASSERT_NE(restored_temperature, nullptr);

    const atlas::Vector3 restored_p0 = restored_position->data()[0];
    const atlas::Vector3 restored_v1 = restored_velocity->data()[1];

    EXPECT_FLOAT_EQ(restored_p0.x, 1.0f);
    EXPECT_FLOAT_EQ(restored_p0.y, 2.0f);
    EXPECT_FLOAT_EQ(restored_p0.z, 3.0f);
    EXPECT_FLOAT_EQ(restored_v1.x, 0.4f);
    EXPECT_FLOAT_EQ(restored_v1.y, 0.5f);
    EXPECT_FLOAT_EQ(restored_v1.z, 0.6f);
    EXPECT_EQ(restored_active->data()[0], 1);
    EXPECT_FLOAT_EQ(restored_temperature->data()[1], 450.0f);

    std::filesystem::remove(snapshot_path);
}
