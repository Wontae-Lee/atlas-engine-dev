#include "../utilities/test_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

#include <filesystem>
#include <memory>

namespace {

namespace fs = std::filesystem;

using atlas::DeviceBuffer;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::Vector3F;
using atlas::test::vec_near;
using atlas::tol;
using atlas::Fluid;
using atlas::FluidActiveState;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidTemperatureState;
using atlas::FluidVelocityState;
using atlas::load_fluid_binary;
using atlas::load_universe_binary;
using atlas::save_fluid_binary;
using atlas::save_universe_binary;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::Universe;
using atlas::UniverseBulkVelocityState;
using atlas::UniverseCollisionCountState;
using atlas::UniverseKnudsenNumberState;
using atlas::UniverseTemperatureState;

using FloatFluid = Fluid<float>;
using FloatGeneratorHostPtr = GeneratorHostPtr<float>;
using FloatMaterialProperties = MaterialProperties<float>;
using FloatUniverse = Universe<float>;

} // namespace

TEST(ProtobufSnapshot, SaveAndLoadFluidBinarySnapshotPayload) {
    // Arrange: create a fluid with active particles and serializable states.
    HostBuffer<FloatMaterialProperties> properties(1);
    HostBuffer<FloatGeneratorHostPtr> generators(1);

    properties[0].type = MaterialType::Molecule;
    properties[0].mass = 6.0f;
    properties[0].molecular_mass = 2.0f;
    properties[0].species_id = 7;
    properties[0].reference_diameter = 4.0f;
    properties[0].reference_temperature = 273.15f;

    auto fluid = FloatFluid::builder()
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

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_protobuf_fluid_snapshot_test.bin";

    // Act: save and reload the fluid snapshot.
    save_fluid_binary(fluid, snapshot_path.string());
    const auto snapshot = load_fluid_binary<float>(snapshot_path.string());

    // Assert: snapshot metadata and state payloads round-trip.
    EXPECT_EQ(snapshot.buffer_size, 4u);
    EXPECT_EQ(snapshot.particle_count, 2u);
    EXPECT_FLOAT_EQ(snapshot.statistical_weight, 3.0f);
    ASSERT_EQ(snapshot.properties.size(), 1u);
    EXPECT_EQ(snapshot.properties[0].species_id.value_or(-1), 7);
    EXPECT_FLOAT_EQ(snapshot.properties[0].reference_diameter.value_or(0.0f), 4.0f);
    EXPECT_FLOAT_EQ(snapshot.properties[0].reference_temperature.value_or(0.0f), 273.15f);

    ASSERT_TRUE(snapshot.positions.has_value());
    ASSERT_TRUE(snapshot.velocities.has_value());
    ASSERT_TRUE(snapshot.species.has_value());
    ASSERT_TRUE(snapshot.active.has_value());
    ASSERT_TRUE(snapshot.temperature.has_value());

    EXPECT_TRUE(vec_near((*snapshot.positions)[0], Vector3F(1.0f, 2.0f, 3.0f), tol));
    EXPECT_TRUE(vec_near((*snapshot.velocities)[1], Vector3F(0.4f, 0.5f, 0.6f), tol));
    EXPECT_EQ((*snapshot.species)[1], 0u);
    EXPECT_EQ((*snapshot.active)[0], 1);
    EXPECT_FLOAT_EQ((*snapshot.temperature)[1], 450.0f);

    fs::remove(snapshot_path);
}

TEST(ProtobufSnapshot, SaveAndLoadUniverseBinarySnapshotPayload) {
    // Arrange: create a universe with serializable states.
    auto universe = FloatUniverse::builder()
                        .with_lower_corner(Vector3F(-1.0f, -2.0f, -3.0f))
                        .with_upper_corner(Vector3F(1.0f, 2.0f, 3.0f))
                        .with_cell_size(1.0f)
                        .build();

    universe.set_state<UniverseTemperatureState<float>>(
        std::make_unique<UniverseTemperatureState<float>>(
            DeviceBuffer<float> { 300.0f, 325.0f, 350.0f }));
    universe.set_state<UniverseBulkVelocityState<float>>(
        std::make_unique<UniverseBulkVelocityState<float>>(
            DeviceBuffer<Vector3F> {
                Vector3F(1.0f, 0.0f, 0.0f),
                Vector3F(0.0f, 1.0f, 0.0f),
                Vector3F(0.0f, 0.0f, 1.0f),
            }));
    universe.set_state<UniverseCollisionCountState<int>>(
        std::make_unique<UniverseCollisionCountState<int>>(
            DeviceBuffer<int> { 2, 4, 6 }));
    universe.set_state<UniverseKnudsenNumberState<float>>(
        std::make_unique<UniverseKnudsenNumberState<float>>(
            DeviceBuffer<float> { 0.001f, 0.02f, 0.3f }));

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_protobuf_universe_snapshot_test.bin";

    // Act: save and reload the universe snapshot.
    save_universe_binary(universe, snapshot_path.string());
    const auto snapshot = load_universe_binary<float>(snapshot_path.string());

    // Assert: universe metadata and state payloads round-trip.
    EXPECT_TRUE(vec_near(snapshot.lower_corner, Vector3F(-1.0f, -2.0f, -3.0f), tol));
    EXPECT_TRUE(vec_near(snapshot.upper_corner, Vector3F(1.0f, 2.0f, 3.0f), tol));
    EXPECT_FLOAT_EQ(snapshot.cell_size, 1.0f);

    ASSERT_TRUE(snapshot.temperature.has_value());
    ASSERT_TRUE(snapshot.bulk_velocity.has_value());
    ASSERT_TRUE(snapshot.collision_count.has_value());
    ASSERT_TRUE(snapshot.knudsen_number.has_value());

    EXPECT_FLOAT_EQ((*snapshot.temperature)[1], 325.0f);
    EXPECT_TRUE(vec_near((*snapshot.bulk_velocity)[2], Vector3F(0.0f, 0.0f, 1.0f), tol));
    EXPECT_EQ((*snapshot.collision_count)[0], 2);
    EXPECT_NEAR((*snapshot.knudsen_number)[2], 0.3f, tol);

    fs::remove(snapshot_path);
}
