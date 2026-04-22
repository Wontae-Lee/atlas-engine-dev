#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/serialization/protobuf_snapshot.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

#include <filesystem>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(ProtobufSnapshot, SaveAndLoadFluidBinarySnapshotPayload) {
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

    position->data()[0] = Vec3(1.0f, 2.0f, 3.0f);
    position->data()[1] = Vec3(4.0f, 5.0f, 6.0f);
    velocity->data()[0] = Vec3(0.1f, 0.2f, 0.3f);
    velocity->data()[1] = Vec3(0.4f, 0.5f, 0.6f);
    species->data()[0] = 0u;
    species->data()[1] = 0u;
    active->data()[0] = 1;
    active->data()[1] = 1;
    temperature->data()[0] = 300.0f;
    temperature->data()[1] = 450.0f;

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_protobuf_fluid_snapshot_test.bin";
    atlas::serialization::save_fluid_binary(fluid, snapshot_path.string());

    const auto snapshot = atlas::serialization::load_fluid_binary<T>(snapshot_path.string());

    EXPECT_EQ(snapshot.buffer_size, 4u);
    EXPECT_EQ(snapshot.particle_count, 2u);
    EXPECT_FLOAT_EQ(snapshot.statistical_weight, 3.0f);
    ASSERT_EQ(snapshot.properties.size(), 1u);
    EXPECT_EQ(snapshot.properties[0].species_id.value_or(-1), 7);

    ASSERT_TRUE(snapshot.positions.has_value());
    ASSERT_TRUE(snapshot.velocities.has_value());
    ASSERT_TRUE(snapshot.species.has_value());
    ASSERT_TRUE(snapshot.active.has_value());
    ASSERT_TRUE(snapshot.temperature.has_value());

    EXPECT_TRUE(atlas::test::vec_near((*snapshot.positions)[0], Vec3(1.0f, 2.0f, 3.0f), kEps));
    EXPECT_TRUE(atlas::test::vec_near((*snapshot.velocities)[1], Vec3(0.4f, 0.5f, 0.6f), kEps));
    EXPECT_EQ((*snapshot.species)[1], 0u);
    EXPECT_EQ((*snapshot.active)[0], 1);
    EXPECT_FLOAT_EQ((*snapshot.temperature)[1], 450.0f);

    fs::remove(snapshot_path);
}

TEST(ProtobufSnapshot, SaveAndLoadUniverseBinarySnapshotPayload) {
    namespace fs = std::filesystem;

    auto universe = atlas::universe::Universe<T>::builder()
                        .with_lower_corner(Vec3(-1.0f, -2.0f, -3.0f))
                        .with_upper_corner(Vec3(1.0f, 2.0f, 3.0f))
                        .with_cell_size(1.0f)
                        .build();

    universe.set_state<atlas::universe::UniverseTemperatureState<T>>(
        std::make_unique<atlas::universe::UniverseTemperatureState<T>>(
            atlas::DeviceBuffer<T> { 300.0f, 325.0f, 350.0f }));
    universe.set_state<atlas::universe::UniverseBulkVelocityState<T>>(
        std::make_unique<atlas::universe::UniverseBulkVelocityState<T>>(
            atlas::DeviceBuffer<Vec3> {
                Vec3(1.0f, 0.0f, 0.0f),
                Vec3(0.0f, 1.0f, 0.0f),
                Vec3(0.0f, 0.0f, 1.0f),
            }));
    universe.set_state<atlas::universe::UniverseCollisionCountState<int>>(
        std::make_unique<atlas::universe::UniverseCollisionCountState<int>>(
            atlas::DeviceBuffer<int> { 2, 4, 6 }));
    universe.set_state<atlas::universe::UniverseKnudsenNumberState<T>>(
        std::make_unique<atlas::universe::UniverseKnudsenNumberState<T>>(
            atlas::DeviceBuffer<T> { 0.001f, 0.02f, 0.3f }));

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_protobuf_universe_snapshot_test.bin";
    atlas::serialization::save_universe_binary(universe, snapshot_path.string());

    const auto snapshot = atlas::serialization::load_universe_binary<T>(snapshot_path.string());

    EXPECT_TRUE(atlas::test::vec_near(snapshot.lower_corner, Vec3(-1.0f, -2.0f, -3.0f), kEps));
    EXPECT_TRUE(atlas::test::vec_near(snapshot.upper_corner, Vec3(1.0f, 2.0f, 3.0f), kEps));
    EXPECT_FLOAT_EQ(snapshot.cell_size, 1.0f);

    ASSERT_TRUE(snapshot.temperature.has_value());
    ASSERT_TRUE(snapshot.bulk_velocity.has_value());
    ASSERT_TRUE(snapshot.collision_count.has_value());
    ASSERT_TRUE(snapshot.knudsen_number.has_value());

    EXPECT_FLOAT_EQ((*snapshot.temperature)[1], 325.0f);
    EXPECT_TRUE(atlas::test::vec_near((*snapshot.bulk_velocity)[2], Vec3(0.0f, 0.0f, 1.0f), kEps));
    EXPECT_EQ((*snapshot.collision_count)[0], 2);
    EXPECT_NEAR((*snapshot.knudsen_number)[2], 0.3f, kEps);

    fs::remove(snapshot_path);
}
