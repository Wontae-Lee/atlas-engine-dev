#include "../utilities/test_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

#include <filesystem>

namespace {

using atlas::Box;
using atlas::DeviceBuffer;
using atlas::eps;
using atlas::Universe;
using atlas::Vector3F;
using atlas::Vector3I;
using atlas::test::vec_near;
using atlas::universe::UniverseBulkVelocityState;
using atlas::universe::UniverseCollisionCountState;
using atlas::universe::UniverseTemperatureState;

namespace fs = std::filesystem;

} // namespace

TEST(Universe, BuilderConstructsConfiguredUniverse) {
    // Arrange and act: build a universe with a known cubic domain and cell size.
    const auto universe = Universe<float>::builder()
                              .with_lower_corner(Vector3F(0, 0, 0))
                              .with_upper_corner(Vector3F(1, 1, 1))
                              .with_cell_size(0.5f)
                              .build();

    // Assert: verify that the configured domain bounds are preserved.
    EXPECT_TRUE(vec_near(universe.lower_corner(), Vector3F(0, 0, 0), eps));
    EXPECT_TRUE(vec_near(universe.upper_corner(), Vector3F(1, 1, 1), eps));

    // Assert: [0, 1] with h = 0.5 produces 3 grid points per axis.
    EXPECT_TRUE(vec_near(universe.grid_size(), Vector3I(3, 3, 3), 0));
    EXPECT_EQ(universe.number_of_cells(), 27);

    // Assert: verify derived cell metrics.
    EXPECT_FLOAT_EQ(universe.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(universe.cell_volume(), 0.125f);
    EXPECT_FLOAT_EQ(universe.inverse_cell_size(), 2.0f);
}

TEST(Universe, BuilderCanUseGeometryBounds) {
    // Prepare geometry bounds for the universe builder.
    const auto geometry = Box<float>::builder()
                              .with_lower_corner(Vector3F(-1, -2, -3))
                              .with_upper_corner(Vector3F(1, 2, 3))
                              .make_host_shared();

    const auto universe = Universe<float>::builder()
                              .with_geometry(geometry)
                              .with_cell_size(1.0f)
                              .build();

    // The universe should inherit the geometry bounds.
    EXPECT_TRUE(vec_near(universe.lower_corner(), Vector3F(-1, -2, -3), eps));
    EXPECT_TRUE(vec_near(universe.upper_corner(), Vector3F(1, 2, 3), eps));
}
TEST(Universe, BuilderRejectsInvalidGeometryParameters) {
    // A zero cell size cannot define a valid grid.
    EXPECT_THROW(Universe<float>::builder()
                     .with_lower_corner(Vector3F(0, 0, 0))
                     .with_upper_corner(Vector3F(1, 1, 1))
                     .with_cell_size(0.0f)
                     .build(),
                 std::invalid_argument);

    // The upper corner must be greater than the lower corner on every axis.
    EXPECT_THROW(Universe<float>::builder()
                     .with_lower_corner(Vector3F(1, 1, 1))
                     .with_upper_corner(Vector3F(0, 0, 0))
                     .with_cell_size(1.0f)
                     .build(),
                 std::invalid_argument);
}

TEST(Universe, StateLifecycleSupportsInsertLookupReplaceAndRemove) {
    // Build a minimal universe for testing state ownership.
    auto universe = Universe<float>::builder()
                        .with_lower_corner(Vector3F(0, 0, 0))
                        .with_upper_corner(Vector3F(1, 1, 1))
                        .with_cell_size(1.0f)
                        .build();

    // The temperature state should not exist before insertion.
    EXPECT_FALSE(universe.has_state<UniverseTemperatureState<float>>());

    // Insert a new state and verify that it can be found.
    auto& state = universe.emplace_state<UniverseTemperatureState<float>>(8);
    EXPECT_EQ(state.size(), 8u);
    EXPECT_TRUE(universe.has_state<UniverseTemperatureState<float>>());

    // Replace the existing state with a new instance.
    universe.set_state<UniverseTemperatureState<float>>(
        std::make_unique<UniverseTemperatureState<float>>(4));

    // The lookup should return the replacement state.
    ASSERT_NE(universe.state<UniverseTemperatureState<float>>(), nullptr);
    EXPECT_EQ(universe.state<UniverseTemperatureState<float>>()->size(), 4u);

    // Removing the state should transfer ownership back to the caller.
    auto removed = universe.remove_state<UniverseTemperatureState<float>>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), 4u);
    EXPECT_FALSE(universe.has_state<UniverseTemperatureState<float>>());
}

TEST(Universe, SetStateRejectsNullOwnershipTransfer) {
    // Build a minimal universe for testing invalid state assignment.
    auto universe = Universe<float>::builder()
                        .with_lower_corner(Vector3F(0, 0, 0))
                        .with_upper_corner(Vector3F(1, 1, 1))
                        .with_cell_size(1.0f)
                        .build();

    std::unique_ptr<UniverseTemperatureState<float>> null_state;

    // set_state must reject null ownership transfers.
    EXPECT_THROW(universe.set_state<UniverseTemperatureState<float>>(std::move(null_state)),
                 std::invalid_argument);
}

TEST(Universe, SaveAndReloadBinarySnapshot) {
    // Build a universe with deterministic geometry for serialization.
    auto universe = Universe<float>::builder()
                        .with_lower_corner(Vector3F(-1.0f, -2.0f, -3.0f))
                        .with_upper_corner(Vector3F(1.0f, 2.0f, 3.0f))
                        .with_cell_size(1.0f)
                        .build();

    // Attach representative scalar, vector, and integer states.
    universe.set_state<UniverseTemperatureState<float>>(
        std::make_unique<UniverseTemperatureState<float>>(
            DeviceBuffer<float> { 300.0f, 325.0f, 350.0f }));

    universe.set_state<UniverseBulkVelocityState<float>>(
        std::make_unique<UniverseBulkVelocityState<float>>(
            DeviceBuffer<Vector3F> {
                Vector3F(1.0f, 0.0f, 0.0f),
                Vector3F(0.0f, 1.0f, 0.0f),
                Vector3F(0.0f, 0.0f, 1.0f) }));

    universe.set_state<UniverseCollisionCountState<int>>(
        std::make_unique<UniverseCollisionCountState<int>>(
            DeviceBuffer<int> { 2, 4, 6 }));

    // Save the snapshot to a temporary binary file.
    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_universe_snapshot_test.bin";
    universe.save(snapshot_path.string());

    // Rebuild a universe from the saved snapshot.
    const auto restored = Universe<float>::builder()
                              .with_binary(snapshot_path.string())
                              .build();

    // Verify that the serialized geometry was restored.
    EXPECT_TRUE(vec_near(restored.lower_corner(), Vector3F(-1.0f, -2.0f, -3.0f), eps));
    EXPECT_TRUE(vec_near(restored.upper_corner(), Vector3F(1.0f, 2.0f, 3.0f), eps));
    EXPECT_FLOAT_EQ(restored.cell_size(), 1.0f);

    // Retrieve restored states by concrete type.
    const auto* temperature     = restored.state<UniverseTemperatureState<float>>();
    const auto* bulk_velocity   = restored.state<UniverseBulkVelocityState<float>>();
    const auto* collision_count = restored.state<UniverseCollisionCountState<int>>();

    ASSERT_NE(temperature, nullptr);
    ASSERT_NE(bulk_velocity, nullptr);
    ASSERT_NE(collision_count, nullptr);

    // Verify representative restored state values.
    EXPECT_FLOAT_EQ(temperature->data()[1], 325.0f);
    EXPECT_TRUE(vec_near(bulk_velocity->data()[2], Vector3F(0.0f, 0.0f, 1.0f), eps));
    EXPECT_EQ(collision_count->data()[0], 2);
    EXPECT_EQ(collision_count->data()[2], 6);

    // Clean up the temporary snapshot file.
    fs::remove(snapshot_path);
}