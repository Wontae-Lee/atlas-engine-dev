#include <atlas/universe/universe.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/observer/observer.h>

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

bool
expect_vec_near(const atlas::Float3& a, const atlas::Float3& b) {
    return std::abs(a.x - b.x) <= atlas::tol
        && std::abs(a.y - b.y) <= atlas::tol
        && std::abs(a.z - b.z) <= atlas::tol;
}

}

TEST(Universe, BuilderConstructsConfiguredUniverse) {
    const auto universe = atlas::Universe::builder()
                              .with_lower_corner(atlas::Float3(0.0f, 0.0f, 0.0f))
                              .with_upper_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
                              .with_cell_size(0.5f)
                              .build();

    EXPECT_TRUE(expect_vec_near(universe.lower_corner(), atlas::Float3(0.0f, 0.0f, 0.0f)));
    EXPECT_TRUE(expect_vec_near(universe.upper_corner(), atlas::Float3(1.0f, 1.0f, 1.0f)));

    EXPECT_TRUE(universe.grid_size() == atlas::Int3(3, 3, 3));
    EXPECT_EQ(universe.cell_count(), 27);

    EXPECT_FLOAT_EQ(universe.cell_size(), 0.5f);
    EXPECT_FLOAT_EQ(universe.cell_volume(), 0.125f);
    EXPECT_FLOAT_EQ(universe.inverse_cell_size(), 2.0f);
}

TEST(Universe, BuilderCanUseGeometryBounds) {
    const atlas::Geometry geometry(atlas::Box::builder()
                                               .with_lower_corner(atlas::Float3(-1.0f, -2.0f, -3.0f))
                                               .with_upper_corner(atlas::Float3(1.0f, 2.0f, 3.0f))
                                               .build());

    const auto universe = atlas::Universe::builder()
                              .with_geometry(geometry)
                              .with_cell_size(1.0f)
                              .build();

    EXPECT_TRUE(expect_vec_near(universe.lower_corner(), atlas::Float3(-1.0f, -2.0f, -3.0f)));
    EXPECT_TRUE(expect_vec_near(universe.upper_corner(), atlas::Float3(1.0f, 2.0f, 3.0f)));
}

TEST(Universe, BuilderRejectsInvalidGeometryParameters) {
    EXPECT_THROW(static_cast<void>(atlas::Universe::builder()
                                       .with_lower_corner(atlas::Float3(0.0f, 0.0f, 0.0f))
                                       .with_upper_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
                                       .with_cell_size(0.0f)
                                       .build()),
                 std::invalid_argument);

    EXPECT_THROW(static_cast<void>(atlas::Universe::builder()
                                       .with_lower_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
                                       .with_upper_corner(atlas::Float3(0.0f, 0.0f, 0.0f))
                                       .with_cell_size(1.0f)
                                       .build()),
                 std::invalid_argument);
}

TEST(Universe, BuilderAttachesObserver) {
    auto observer = atlas::Observer::builder().make_host_shared();

    const auto universe = atlas::Universe::builder()
                              .with_lower_corner(atlas::Float3(0.0f, 0.0f, 0.0f))
                              .with_upper_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
                              .with_cell_size(1.0f)
                              .with_observer(observer)
                              .build();

    EXPECT_EQ(universe.observer().get(), observer.get());
}

TEST(Universe, StateLifecycleSupportsInsertLookupReplaceAndRemove) {
    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(0.0f, 0.0f, 0.0f))
                        .with_upper_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
                        .with_cell_size(1.0f)
                        .build();

    EXPECT_FALSE(universe.has_state<atlas::UniverseTemperatureState>());

    auto& state = universe.emplace_state<atlas::UniverseTemperatureState>(8);
    EXPECT_EQ(state.size(), 8u);
    EXPECT_TRUE(universe.has_state<atlas::UniverseTemperatureState>());

    universe.set_state<atlas::UniverseTemperatureState>(
        std::make_unique<atlas::UniverseTemperatureState>(4));

    ASSERT_NE(universe.state<atlas::UniverseTemperatureState>(), nullptr);
    EXPECT_EQ(universe.state<atlas::UniverseTemperatureState>()->size(), 4u);

    auto removed = universe.remove_state<atlas::UniverseTemperatureState>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), 4u);
    EXPECT_FALSE(universe.has_state<atlas::UniverseTemperatureState>());
}

TEST(Universe, SetStateRejectsNullOwnershipTransfer) {
    auto universe = atlas::Universe::builder()
                        .with_lower_corner(atlas::Float3(0.0f, 0.0f, 0.0f))
                        .with_upper_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
                        .with_cell_size(1.0f)
                        .build();

    std::unique_ptr<atlas::UniverseTemperatureState> null_state;

    EXPECT_THROW(universe.set_state<atlas::UniverseTemperatureState>(std::move(null_state)),
                 std::invalid_argument);
}

TEST(Universe, SaveAndReloadBinarySnapshot) {
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
                atlas::Float3(0.0f, 0.0f, 1.0f) }));

    universe.set_state<atlas::UniverseCollisionCountState>(
        std::make_unique<atlas::UniverseCollisionCountState>(
            atlas::DeviceBuffer<int> { 2, 4, 6 }));

    const std::filesystem::path snapshot_path
        = std::filesystem::temp_directory_path() / "atlas_universe_snapshot_test.bin";
    universe.save(snapshot_path.string());

    const auto restored = atlas::Universe::builder()
                              .with_binary(snapshot_path.string())
                              .build();

    EXPECT_TRUE(expect_vec_near(restored.lower_corner(), atlas::Float3(-1.0f, -2.0f, -3.0f)));
    EXPECT_TRUE(expect_vec_near(restored.upper_corner(), atlas::Float3(1.0f, 2.0f, 3.0f)));
    EXPECT_FLOAT_EQ(restored.cell_size(), 1.0f);

    const auto* temperature     = restored.state<atlas::UniverseTemperatureState>();
    const auto* bulk_velocity   = restored.state<atlas::UniverseBulkVelocityState>();
    const auto* collision_count = restored.state<atlas::UniverseCollisionCountState>();

    ASSERT_NE(temperature, nullptr);
    ASSERT_NE(bulk_velocity, nullptr);
    ASSERT_NE(collision_count, nullptr);

    const float restored_t1          = temperature->data()[1];
    const atlas::Float3 restored_v2 = bulk_velocity->data()[2];
    const int restored_c0            = collision_count->data()[0];
    const int restored_c2            = collision_count->data()[2];

    EXPECT_FLOAT_EQ(restored_t1, 325.0f);
    EXPECT_TRUE(expect_vec_near(restored_v2, atlas::Float3(0.0f, 0.0f, 1.0f)));
    EXPECT_EQ(restored_c0, 2);
    EXPECT_EQ(restored_c2, 6);

    std::filesystem::remove(snapshot_path);
}
