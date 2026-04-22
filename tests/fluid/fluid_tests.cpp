#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>

#include <testkit/testkit.h>

#include <filesystem>

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

TEST(Fluid, SetParticleCountTracksActivePrefix) {
    atlas::fluid::Fluid<T> fluid(4);

    EXPECT_NO_THROW(fluid.set_particle_count(3));
    EXPECT_EQ(fluid.particle_count(), 3u);

    EXPECT_THROW(fluid.set_particle_count(5), std::out_of_range);
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

TEST(Fluid, SaveAndReloadBinarySnapshot) {
    namespace fs = std::filesystem;

    atlas::HostBuffer<atlas::system::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0].type = atlas::system::MaterialType::Molecule;
    properties[0].mass = 6.0f;
    properties[0].molecular_mass = 2.0f;
    properties[0].species_id = 7;

    auto fluid = atlas::fluid::Fluid<T>::builder()
                     .with_buffer_size(4)
                     .with_statistical_weight(3.0f)
                     .with_properties(properties)
                     .with_generators(generators)
                     .build();

    fluid.emplace_state<atlas::fluid::FluidTemperatureState<T>>(4);
    fluid.set_particle_count(2);

    auto* position = fluid.state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity = fluid.state<atlas::fluid::FluidVelocityState<T>>();
    auto* species = fluid.state<atlas::fluid::FluidSpeciesState<T>>();
    auto* active = fluid.state<atlas::fluid::FluidActiveState<T>>();
    auto* temperature = fluid.state<atlas::fluid::FluidTemperatureState<T>>();

    ASSERT_NE(position, nullptr);
    ASSERT_NE(velocity, nullptr);
    ASSERT_NE(species, nullptr);
    ASSERT_NE(active, nullptr);
    ASSERT_NE(temperature, nullptr);

    position->data()[0] = atlas::Vector3<T>(1.0f, 2.0f, 3.0f);
    position->data()[1] = atlas::Vector3<T>(4.0f, 5.0f, 6.0f);
    velocity->data()[0] = atlas::Vector3<T>(0.1f, 0.2f, 0.3f);
    velocity->data()[1] = atlas::Vector3<T>(0.4f, 0.5f, 0.6f);
    species->data()[0] = 0u;
    species->data()[1] = 0u;
    active->data()[0] = 1;
    active->data()[1] = 1;
    temperature->data()[0] = 300.0f;
    temperature->data()[1] = 450.0f;

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_fluid_snapshot_test.bin";
    fluid.save(snapshot_path.string());

    const auto restored = atlas::fluid::Fluid<T>::builder()
                              .with_binary(snapshot_path.string())
                              .build();

    EXPECT_EQ(restored.buffer_size(), 4u);
    EXPECT_EQ(restored.particle_count(), 2u);
    EXPECT_FLOAT_EQ(restored.statistical_weight(), 3.0f);
    ASSERT_EQ(restored.particle_properties().size(), 1u);
    EXPECT_EQ(restored.particle_properties()[0].species_id.value_or(-1), 7);

    const auto* restored_position = restored.state<atlas::fluid::FluidPositionState<T>>();
    const auto* restored_velocity = restored.state<atlas::fluid::FluidVelocityState<T>>();
    const auto* restored_active = restored.state<atlas::fluid::FluidActiveState<T>>();
    const auto* restored_temperature = restored.state<atlas::fluid::FluidTemperatureState<T>>();

    ASSERT_NE(restored_position, nullptr);
    ASSERT_NE(restored_velocity, nullptr);
    ASSERT_NE(restored_active, nullptr);
    ASSERT_NE(restored_temperature, nullptr);

    EXPECT_TRUE(atlas::test::vec_near(restored_position->data()[0], atlas::Vector3<T>(1.0f, 2.0f, 3.0f), 1e-5f));
    EXPECT_TRUE(atlas::test::vec_near(restored_velocity->data()[1], atlas::Vector3<T>(0.4f, 0.5f, 0.6f), 1e-5f));
    EXPECT_EQ(restored_active->data()[0], 1);
    EXPECT_FLOAT_EQ(restored_temperature->data()[1], 450.0f);

    fs::remove(snapshot_path);
}
