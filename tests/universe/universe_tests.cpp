#include "../utilities/tests_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

#include <filesystem>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;
using IVec3 = atlas::Vector3<int>;

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Universe, BuilderConstructsConfiguredUniverse) {
    const auto universe = atlas::universe::Universe<T>::builder()
                              .with_lower_corner(Vec3(0, 0, 0))
                              .with_upper_corner(Vec3(1, 1, 1))
                              .with_cell_size(0.5f)
                              .build();

    EXPECT_TRUE(atlas::test::vec_near(universe.lower_corner(), Vec3(0, 0, 0), kEps));
    EXPECT_TRUE(atlas::test::vec_near(universe.upper_corner(), Vec3(1, 1, 1), kEps));
    EXPECT_TRUE(atlas::test::vec_near(universe.grid_size(), IVec3(3, 3, 3), 0));
    EXPECT_EQ(universe.number_of_cells(), 27);
    EXPECT_FLOAT_EQ(universe.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(universe.cell_volume(), 0.125f);
    EXPECT_FLOAT_EQ(universe.inverse_cell_size(), 2.0f);
}

TEST(Universe, BuilderCanUseGeometryBounds) {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(Vec3(-1, -2, -3))
                              .with_upper_corner(Vec3(1, 2, 3))
                              .make_host_shared();

    const auto universe = atlas::universe::Universe<T>::builder()
                              .with_geometry(geometry)
                              .with_cell_size(1.0f)
                              .build();

    EXPECT_TRUE(atlas::test::vec_near(universe.lower_corner(), Vec3(-1, -2, -3), kEps));
    EXPECT_TRUE(atlas::test::vec_near(universe.upper_corner(), Vec3(1, 2, 3), kEps));
}

TEST(Universe, BuilderRejectsInvalidGeometryParameters) {
    EXPECT_THROW(
        atlas::universe::Universe<T>::builder()
            .with_lower_corner(Vec3(0, 0, 0))
            .with_upper_corner(Vec3(1, 1, 1))
            .with_cell_size(0.0f)
            .build(),
        std::invalid_argument);

    EXPECT_THROW(
        atlas::universe::Universe<T>::builder()
            .with_lower_corner(Vec3(1, 1, 1))
            .with_upper_corner(Vec3(0, 0, 0))
            .with_cell_size(1.0f)
            .build(),
        std::invalid_argument);
}

TEST(Universe, StateLifecycleSupportsInsertLookupReplaceAndRemove) {
    auto universe = atlas::universe::Universe<T>::builder()
                        .with_lower_corner(Vec3(0, 0, 0))
                        .with_upper_corner(Vec3(1, 1, 1))
                        .with_cell_size(1.0f)
                        .build();

    EXPECT_FALSE(universe.has_state<atlas::universe::UniverseTemperatureState<T>>());

    auto& state = universe.emplace_state<atlas::universe::UniverseTemperatureState<T>>(8);
    EXPECT_EQ(state.size(), 8u);
    EXPECT_TRUE(universe.has_state<atlas::universe::UniverseTemperatureState<T>>());

    universe.set_state<atlas::universe::UniverseTemperatureState<T>>(std::make_unique<atlas::universe::UniverseTemperatureState<T>>(4));
    ASSERT_NE(universe.state<atlas::universe::UniverseTemperatureState<T>>(), nullptr);
    EXPECT_EQ(universe.state<atlas::universe::UniverseTemperatureState<T>>()->size(), 4u);

    auto removed = universe.remove_state<atlas::universe::UniverseTemperatureState<T>>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), 4u);
    EXPECT_FALSE(universe.has_state<atlas::universe::UniverseTemperatureState<T>>());
}

TEST(Universe, SetStateRejectsNullOwnershipTransfer) {
    auto universe = atlas::universe::Universe<T>::builder()
                        .with_lower_corner(Vec3(0, 0, 0))
                        .with_upper_corner(Vec3(1, 1, 1))
                        .with_cell_size(1.0f)
                        .build();

    std::unique_ptr<atlas::universe::UniverseTemperatureState<T>> null_state;

    EXPECT_THROW(universe.set_state<atlas::universe::UniverseTemperatureState<T>>(std::move(null_state)), std::invalid_argument);
}

TEST(Universe, SaveAndReloadBinarySnapshot) {
    namespace fs = std::filesystem;

    auto universe = atlas::universe::Universe<T>::builder()
                        .with_lower_corner(Vec3(-1.0f, -2.0f, -3.0f))
                        .with_upper_corner(Vec3(1.0f, 2.0f, 3.0f))
                        .with_cell_size(1.0f)
                        .build();

    universe.set_state<atlas::universe::UniverseTemperatureState<T>>(
        std::make_unique<atlas::universe::UniverseTemperatureState<T>>(atlas::DeviceBuffer<T> { 300.0f, 325.0f, 350.0f }));
    universe.set_state<atlas::universe::UniverseBulkVelocityState<T>>(
        std::make_unique<atlas::universe::UniverseBulkVelocityState<T>>(atlas::DeviceBuffer<Vec3> {
            Vec3(1.0f, 0.0f, 0.0f),
            Vec3(0.0f, 1.0f, 0.0f),
            Vec3(0.0f, 0.0f, 1.0f) }));
    universe.set_state<atlas::universe::UniverseCollisionCountState<int>>(
        std::make_unique<atlas::universe::UniverseCollisionCountState<int>>(atlas::DeviceBuffer<int> { 2, 4, 6 }));

    const fs::path snapshot_path = fs::temp_directory_path() / "atlas_universe_snapshot_test.bin";
    universe.save(snapshot_path.string());

    const auto restored = atlas::universe::Universe<T>::builder()
                              .with_binary(snapshot_path.string())
                              .build();

    EXPECT_TRUE(atlas::test::vec_near(restored.lower_corner(), Vec3(-1.0f, -2.0f, -3.0f), kEps));
    EXPECT_TRUE(atlas::test::vec_near(restored.upper_corner(), Vec3(1.0f, 2.0f, 3.0f), kEps));
    EXPECT_FLOAT_EQ(restored.cell_size(), 1.0f);

    const auto* temperature = restored.state<atlas::universe::UniverseTemperatureState<T>>();
    const auto* bulk_velocity = restored.state<atlas::universe::UniverseBulkVelocityState<T>>();
    const auto* collision_count = restored.state<atlas::universe::UniverseCollisionCountState<int>>();

    ASSERT_NE(temperature, nullptr);
    ASSERT_NE(bulk_velocity, nullptr);
    ASSERT_NE(collision_count, nullptr);

    EXPECT_FLOAT_EQ(temperature->data()[1], 325.0f);
    EXPECT_TRUE(atlas::test::vec_near(bulk_velocity->data()[2], Vec3(0.0f, 0.0f, 1.0f), kEps));
    EXPECT_EQ(collision_count->data()[0], 2);
    EXPECT_EQ(collision_count->data()[2], 6);

    fs::remove(snapshot_path);
}
