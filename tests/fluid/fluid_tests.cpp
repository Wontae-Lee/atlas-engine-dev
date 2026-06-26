#include "../utilities/test_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>

#include <testkit/testkit.h>

#include <filesystem>

namespace {

namespace fs = std::filesystem;

using atlas::Fluid;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Vector3F;
using atlas::FluidActiveState;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidTemperatureState;
using atlas::FluidVelocityState;
using atlas::test::vec_near;
using atlas::tol;

} // namespace

TEST(Fluid, DefaultConstructorStartsEmpty) {
    const Fluid<float> fluid;

    EXPECT_EQ(fluid.buffer_size(), 0u);
    EXPECT_EQ(fluid.particle_count(), 0u);
    EXPECT_FALSE(fluid.has_state<FluidPositionState<float>>());
    EXPECT_FALSE(fluid.has_state<FluidVelocityState<float>>());
    EXPECT_FALSE(fluid.has_state<FluidSpeciesState<float>>());
    EXPECT_FALSE(fluid.has_state<FluidActiveState<float>>());
}

TEST(Fluid, SizedConstructorRegistersDefaultStates) {
    const Fluid<float> fluid(8);

    ASSERT_TRUE(fluid.has_state<FluidPositionState<float>>());
    ASSERT_TRUE(fluid.has_state<FluidVelocityState<float>>());
    ASSERT_TRUE(fluid.has_state<FluidSpeciesState<float>>());
    ASSERT_TRUE(fluid.has_state<FluidActiveState<float>>());
    EXPECT_EQ(fluid.buffer_size(), 8u);
    EXPECT_EQ(fluid.particle_count(), 0u);
    EXPECT_EQ(fluid.state<FluidPositionState<float>>()->size(), 8u);
    EXPECT_EQ(fluid.state<FluidVelocityState<float>>()->size(), 8u);
    EXPECT_EQ(fluid.state<FluidSpeciesState<float>>()->size(), 8u);
    EXPECT_EQ(fluid.state<FluidActiveState<float>>()->size(), 8u);
}

TEST(Fluid, BuilderConstructsConfiguredFluid) {
    HostBuffer<MaterialProperties<float>> properties(1);
    HostBuffer<GeneratorHostPtr<float>> generators(1);
    properties[0].mass = 10.0f;
    properties[0].molecular_mass = 2.0f;

    const auto fluid = Fluid<float>::builder()
                           .with_buffer_size(16)
                           .with_statistical_weight(5.0f)
                           .with_properties(properties)
                           .with_generators(generators)
                           .build();

    EXPECT_EQ(fluid.buffer_size(), 16u);
    EXPECT_EQ(fluid.generators().size(), 1u);
    EXPECT_FLOAT_EQ(fluid.statistical_weight(), 5.0f);
    EXPECT_TRUE(fluid.has_state<FluidPositionState<float>>());
    EXPECT_TRUE(fluid.has_state<FluidVelocityState<float>>());
    EXPECT_TRUE(fluid.has_state<FluidSpeciesState<float>>());
    EXPECT_TRUE(fluid.has_state<FluidActiveState<float>>());
}

TEST(Fluid, BuilderRejectsMismatchedPropertiesAndGenerators) {
    HostBuffer<MaterialProperties<float>> properties(1);
    HostBuffer<GeneratorHostPtr<float>> generators(0);

    EXPECT_THROW(
        Fluid<float>::builder()
            .with_buffer_size(4)
            .with_properties(properties)
            .with_generators(generators)
            .build(),
        std::runtime_error);
}

TEST(Fluid, BuilderRejectsNonPositiveStatisticalWeight) {
    HostBuffer<MaterialProperties<float>> properties(0);
    HostBuffer<GeneratorHostPtr<float>> generators(0);

    EXPECT_THROW(
        Fluid<float>::builder()
            .with_buffer_size(4)
            .with_statistical_weight(0.0f)
            .with_properties(properties)
            .with_generators(generators)
            .build(),
        std::runtime_error);
}

TEST(Fluid, BuilderRejectsMassThatDoesNotMatchStatisticalWeight) {
    HostBuffer<MaterialProperties<float>> properties(1);
    HostBuffer<GeneratorHostPtr<float>> generators(1);
    properties[0].mass = 9.0f;
    properties[0].molecular_mass = 2.0f;

    EXPECT_THROW(
        Fluid<float>::builder()
            .with_buffer_size(4)
            .with_statistical_weight(5.0f)
            .with_properties(properties)
            .with_generators(generators)
            .build(),
        std::runtime_error);
}

TEST(Fluid, SetParticleCountTracksActivePrefix) {
    Fluid<float> fluid(4);

    EXPECT_NO_THROW(fluid.set_particle_count(3));
    EXPECT_EQ(fluid.particle_count(), 3u);

    EXPECT_THROW(fluid.set_particle_count(5), std::out_of_range);
}

TEST(Fluid, StateLifecycleSupportsInsertLookupReplaceAndRemove) {
    Fluid<float> fluid(4);

    EXPECT_FALSE(fluid.has_state<FluidTemperatureState<float>>());

    auto& state = fluid.emplace_state<FluidTemperatureState<float>>(4);
    EXPECT_TRUE(fluid.has_state<FluidTemperatureState<float>>());
    EXPECT_EQ(state.size(), 4u);
    ASSERT_NE(fluid.state<FluidTemperatureState<float>>(), nullptr);

    auto replacement = std::make_unique<FluidTemperatureState<float>>(6);
    fluid.set_state<FluidTemperatureState<float>>(std::move(replacement));

    ASSERT_NE(fluid.state<FluidTemperatureState<float>>(), nullptr);
    EXPECT_EQ(fluid.state<FluidTemperatureState<float>>()->size(), 6u);

    auto removed = fluid.remove_state<FluidTemperatureState<float>>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), 6u);
    EXPECT_FALSE(fluid.has_state<FluidTemperatureState<float>>());
    EXPECT_EQ(fluid.state<FluidTemperatureState<float>>(), nullptr);
}

TEST(Fluid, SetStateRejectsNullOwnershipTransfer) {
    Fluid<float> fluid(4);

    std::unique_ptr<FluidTemperatureState<float>> null_state;

    EXPECT_THROW(fluid.set_state<FluidTemperatureState<float>>(std::move(null_state)), std::invalid_argument);
}

TEST(Fluid, SaveAndReloadBinarySnapshot) {
    HostBuffer<MaterialProperties<float>> properties(1);
    HostBuffer<GeneratorHostPtr<float>> generators(1);

    properties[0].type = MaterialType::Molecule;
    properties[0].mass = 6.0f;
    properties[0].molecular_mass = 2.0f;
    properties[0].species_id = 7;

    auto fluid = Fluid<float>::builder()
                     .with_buffer_size(4)
                     .with_statistical_weight(3.0f)
                     .with_properties(properties)
                     .with_generators(generators)
                     .build();

    fluid.emplace_state<FluidTemperatureState<float>>(4);
    fluid.set_particle_count(2);

    auto* position = fluid.state<FluidPositionState<float>>();
    auto* velocity = fluid.state<FluidVelocityState<float>>();
    auto* species = fluid.state<FluidSpeciesState<float>>();
    auto* active = fluid.state<FluidActiveState<float>>();
    auto* temperature = fluid.state<FluidTemperatureState<float>>();

    ASSERT_NE(position, nullptr);
    ASSERT_NE(velocity, nullptr);
    ASSERT_NE(species, nullptr);
    ASSERT_NE(active, nullptr);
    ASSERT_NE(temperature, nullptr);

    position->data()[0] = Vector3F(1.0f, 2.0f, 3.0f);
    position->data()[1] = Vector3F(4.0f, 5.0f, 6.0f);
    velocity->data()[0] = Vector3F(0.1f, 0.2f, 0.3f);
    velocity->data()[1] = Vector3F(0.4f, 0.5f, 0.6f);
    species->data()[0] = 0u;
    species->data()[1] = 0u;
    active->data()[0] = 1;
    active->data()[1] = 1;
    temperature->data()[0] = 300.0f;
    temperature->data()[1] = 450.0f;

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_fluid_snapshot_test.bin";
    fluid.save(snapshot_path.string());

    const auto restored = Fluid<float>::builder()
                              .with_binary(snapshot_path.string())
                              .build();

    EXPECT_EQ(restored.buffer_size(), 4u);
    EXPECT_EQ(restored.particle_count(), 2u);
    EXPECT_FLOAT_EQ(restored.statistical_weight(), 3.0f);
    ASSERT_EQ(restored.particle_properties().size(), 1u);
    const HostBuffer<MaterialProperties<float>> restored_properties(
        restored.particle_properties().begin(),
        restored.particle_properties().end());
    EXPECT_EQ(restored_properties[0].species_id.value_or(-1), 7);

    const auto* restored_position = restored.state<FluidPositionState<float>>();
    const auto* restored_velocity = restored.state<FluidVelocityState<float>>();
    const auto* restored_active = restored.state<FluidActiveState<float>>();
    const auto* restored_temperature = restored.state<FluidTemperatureState<float>>();

    ASSERT_NE(restored_position, nullptr);
    ASSERT_NE(restored_velocity, nullptr);
    ASSERT_NE(restored_active, nullptr);
    ASSERT_NE(restored_temperature, nullptr);

    EXPECT_TRUE(vec_near(restored_position->data()[0], Vector3F(1.0f, 2.0f, 3.0f), tol));
    EXPECT_TRUE(vec_near(restored_velocity->data()[1], Vector3F(0.4f, 0.5f, 0.6f), tol));
    EXPECT_EQ(restored_active->data()[0], 1);
    EXPECT_FLOAT_EQ(restored_temperature->data()[1], 450.0f);

    fs::remove(snapshot_path);
}
