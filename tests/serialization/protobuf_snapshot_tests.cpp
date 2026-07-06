#include <atlas/serialization/protobuf_snapshot.h>

#include <atlas/fluid/fluid.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>

namespace {

bool
expect_vec_near(const atlas::Float3& a, const atlas::Float3& b) {
    return std::abs(a.x - b.x) <= atlas::tol
        && std::abs(a.y - b.y) <= atlas::tol
        && std::abs(a.z - b.z) <= atlas::tol;
}

}

TEST(ProtobufSnapshot, SaveAndLoadFluidBinarySnapshotPayload) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);

    properties[0].type                  = atlas::MaterialType::molecule;
    properties[0].mass                  = 6.0f;
    properties[0].molecular_mass        = 2.0f;
    properties[0].species_id            = 7;
    properties[0].reference_diameter    = 4.0f;
    properties[0].reference_temperature = 273.15f;

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

    position->data()[0]    = atlas::Float3(1.0f, 2.0f, 3.0f);
    position->data()[1]    = atlas::Float3(4.0f, 5.0f, 6.0f);
    velocity->data()[0]    = atlas::Float3(0.1f, 0.2f, 0.3f);
    velocity->data()[1]    = atlas::Float3(0.4f, 0.5f, 0.6f);
    species->data()[0]     = 0u;
    species->data()[1]     = 0u;
    active->data()[0]      = 1;
    active->data()[1]      = 1;
    temperature->data()[0] = 300.0f;
    temperature->data()[1] = 450.0f;

    const std::filesystem::path snapshot_path
        = std::filesystem::temp_directory_path() / "atlas_protobuf_fluid_snapshot_test.bin";

    atlas::save_fluid_binary(fluid, snapshot_path.string());
    const auto snapshot = atlas::load_fluid_binary(snapshot_path.string());

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

    EXPECT_TRUE(expect_vec_near((*snapshot.positions)[0], atlas::Float3(1.0f, 2.0f, 3.0f)));
    EXPECT_TRUE(expect_vec_near((*snapshot.velocities)[1], atlas::Float3(0.4f, 0.5f, 0.6f)));
    EXPECT_EQ((*snapshot.species)[1], 0u);
    EXPECT_EQ((*snapshot.active)[0], 1);
    EXPECT_FLOAT_EQ((*snapshot.temperature)[1], 450.0f);

    std::filesystem::remove(snapshot_path);
}

TEST(ProtobufSnapshot, SaveAndLoadUniverseBinarySnapshotPayload) {
    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(-1.0f, -2.0f, -3.0f))
                        .with_upper_corner(atlas::Float3(1.0f, 2.0f, 3.0f))
                        .with_cell_size(1.0f)
                        .build();

    universe.set_state<atlas::UniverseTemperatureState>(
        std::make_unique<atlas::UniverseTemperatureState>(
            atlas::DeviceBuffer<float> { 300.0f, 325.0f, 350.0f }));
    universe.set_state<atlas::UniverseBulkVelocityState>(
        std::make_unique<atlas::UniverseBulkVelocityState>(
            atlas::DeviceBuffer<atlas::Float3> {
                atlas::Float3(1.0f, 0.0f, 0.0f),
                atlas::Float3(0.0f, 1.0f, 0.0f),
                atlas::Float3(0.0f, 0.0f, 1.0f),
            }));
    universe.set_state<atlas::UniverseCollisionCountState>(
        std::make_unique<atlas::UniverseCollisionCountState>(
            atlas::DeviceBuffer<int> { 2, 4, 6 }));
    universe.set_state<atlas::UniverseKnudsenNumberState>(
        std::make_unique<atlas::UniverseKnudsenNumberState>(
            atlas::DeviceBuffer<float> { 0.001f, 0.02f, 0.3f }));

    const std::filesystem::path snapshot_path
        = std::filesystem::temp_directory_path() / "atlas_protobuf_universe_snapshot_test.bin";

    atlas::save_universe_binary(universe, snapshot_path.string());
    const auto snapshot = atlas::load_universe_binary(snapshot_path.string());

    EXPECT_TRUE(expect_vec_near(snapshot.lower_corner, atlas::Float3(-1.0f, -2.0f, -3.0f)));
    EXPECT_TRUE(expect_vec_near(snapshot.upper_corner, atlas::Float3(1.0f, 2.0f, 3.0f)));
    EXPECT_FLOAT_EQ(snapshot.cell_size, 1.0f);

    ASSERT_TRUE(snapshot.temperature.has_value());
    ASSERT_TRUE(snapshot.bulk_velocity.has_value());
    ASSERT_TRUE(snapshot.collision_count.has_value());
    ASSERT_TRUE(snapshot.knudsen_number.has_value());

    EXPECT_FLOAT_EQ((*snapshot.temperature)[1], 325.0f);
    EXPECT_TRUE(expect_vec_near((*snapshot.bulk_velocity)[2], atlas::Float3(0.0f, 0.0f, 1.0f)));
    EXPECT_EQ((*snapshot.collision_count)[0], 2);
    EXPECT_NEAR((*snapshot.knudsen_number)[2], 0.3f, atlas::tol);

    std::filesystem::remove(snapshot_path);
}
