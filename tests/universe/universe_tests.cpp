#include "../utilities/tests_utils.h"

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

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
