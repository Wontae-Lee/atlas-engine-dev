#include <atlas/universe/universe.h>

#include <atlas/geometry/box.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace {

using atlas::Box;
using atlas::Float3;
using atlas::Geometry;
using atlas::Int3;
using atlas::tol;
using atlas::Universe;
using atlas::UniverseNumberParticleState;

// A box [lower, upper] with the given cell size, built through the fluent path.
Universe
make_universe(const Float3& lower, const Float3& upper, const float cell_size) {
    return Universe::builder()
        .with_lower_corner(lower)
        .with_upper_corner(upper)
        .with_cell_size(cell_size)
        .build();
}

}

TEST(Universe, DefaultBuilderProducesUnitBoxGrid) {
    // The default configuration is the unit box with a unit cell size. With the
    // floor(extent / h) + 1 covering convention that is 2 cells per axis.
    const Universe universe = Universe::builder().build();

    EXPECT_NEAR(universe.cell_size(), 1.0f, tol);
    EXPECT_EQ(universe.grid_size().x, 2);
    EXPECT_EQ(universe.grid_size().y, 2);
    EXPECT_EQ(universe.grid_size().z, 2);
    EXPECT_EQ(universe.cell_count(), 8);
}

TEST(Universe, GridSizeUsesFloorPlusOneCovering) {
    // extent = (3, 2, 1), h = 1 -> floor(extent) + 1 = (4, 3, 2).
    const Universe universe = make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(3.0f, 2.0f, 1.0f), 1.0f);

    EXPECT_EQ(universe.grid_size().x, 4);
    EXPECT_EQ(universe.grid_size().y, 3);
    EXPECT_EQ(universe.grid_size().z, 2);
    EXPECT_EQ(universe.cell_count(), 4 * 3 * 2);
}

TEST(Universe, NonMultipleExtentRoundsUpToCoveringCell) {
    // extent 2.5 with h = 1 -> floor(2.5) + 1 = 3 cells per axis.
    const Universe universe = make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(2.5f, 2.5f, 2.5f), 1.0f);

    EXPECT_EQ(universe.grid_size().x, 3);
    EXPECT_EQ(universe.grid_size().y, 3);
    EXPECT_EQ(universe.grid_size().z, 3);
    EXPECT_EQ(universe.cell_count(), 27);
}

TEST(Universe, ExtentSmallerThanCellStillYieldsOneCell) {
    // extent 0.5 with h = 1 -> floor(0.5) + 1 = 1 cell per axis.
    const Universe universe = make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(0.5f, 0.5f, 0.5f), 1.0f);

    EXPECT_EQ(universe.grid_size().x, 1);
    EXPECT_EQ(universe.grid_size().y, 1);
    EXPECT_EQ(universe.grid_size().z, 1);
    EXPECT_EQ(universe.cell_count(), 1);
}

TEST(Universe, CellVolumeIsCellSizeCubed) {
    const Universe universe = make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 1.0f), 0.5f);

    EXPECT_NEAR(universe.cell_volume(), 0.125f, tol);
    EXPECT_NEAR(universe.inverse_cell_size(), 2.0f, tol);
}

TEST(Universe, CornersRoundTripThroughGetters) {
    const Float3 lower(-1.0f, -2.0f, -3.0f);
    const Float3 upper(4.0f, 5.0f, 6.0f);
    const Universe universe = make_universe(lower, upper, 1.0f);

    EXPECT_NEAR(universe.lower_corner().x, lower.x, tol);
    EXPECT_NEAR(universe.lower_corner().y, lower.y, tol);
    EXPECT_NEAR(universe.lower_corner().z, lower.z, tol);
    EXPECT_NEAR(universe.upper_corner().x, upper.x, tol);
    EXPECT_NEAR(universe.upper_corner().y, upper.y, tol);
    EXPECT_NEAR(universe.upper_corner().z, upper.z, tol);
}

TEST(Universe, BuilderRejectsNonPositiveCellSize) {
    EXPECT_THROW(
        static_cast<void>(Universe::builder().with_cell_size(0.0f).build()),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(Universe::builder().with_cell_size(-1.0f).build()),
        std::invalid_argument);
}

TEST(Universe, BuilderRejectsInvertedDomain) {
    // upper is below lower on the z axis: the strict upper > lower check fails.
    EXPECT_THROW(
        static_cast<void>(make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, -1.0f), 1.0f)),
        std::invalid_argument);
}

TEST(Universe, BuilderRejectsDegenerateDomain) {
    // upper equals lower on the z axis: a zero-extent (zero-resolution) axis is
    // rejected because the corner check is strict, not >=.
    EXPECT_THROW(
        static_cast<void>(make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(1.0f, 1.0f, 0.0f), 1.0f)),
        std::invalid_argument);
}

TEST(Universe, MakeHostUniqueMatchesBuiltGrid) {
    const auto universe = Universe::builder()
                              .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
                              .with_upper_corner(Float3(2.0f, 2.0f, 2.0f))
                              .with_cell_size(1.0f)
                              .make_host_unique();

    ASSERT_TRUE(static_cast<bool>(universe));
    EXPECT_EQ(universe->cell_count(), 27);
}

TEST(Universe, BuilderWithGeometryDerivesDomainFromBound) {
    // with_geometry overwrites both corners from the geometry's axis-aligned bound;
    // the cell size is left at its own setting.
    const Geometry geometry = Geometry(Box::builder()
                                           .with_lower_corner(Float3(-1.0f, -2.0f, -3.0f))
                                           .with_upper_corner(Float3(2.0f, 3.0f, 4.0f))
                                           .build());

    const Universe universe = Universe::builder()
                                  .with_geometry(geometry)
                                  .with_cell_size(1.0f)
                                  .build();

    EXPECT_NEAR(universe.lower_corner().x, -1.0f, tol);
    EXPECT_NEAR(universe.lower_corner().y, -2.0f, tol);
    EXPECT_NEAR(universe.lower_corner().z, -3.0f, tol);
    EXPECT_NEAR(universe.upper_corner().x, 2.0f, tol);
    EXPECT_NEAR(universe.upper_corner().y, 3.0f, tol);
    EXPECT_NEAR(universe.upper_corner().z, 4.0f, tol);
    // extent (3, 5, 7) with h = 1 -> floor(extent) + 1 = (4, 6, 8).
    EXPECT_EQ(universe.cell_count(), 4 * 6 * 8);
}

TEST(Universe, SetStateMovesInAPreBuiltState) {
    Universe universe = make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f), 1.0f);
    const auto cells = static_cast<std::size_t>(universe.cell_count());

    universe.set_state(std::make_unique<UniverseNumberParticleState>(cells));

    ASSERT_TRUE(universe.has_state<UniverseNumberParticleState>());
    ASSERT_NE(universe.state<UniverseNumberParticleState>(), nullptr);
    EXPECT_EQ(universe.state<UniverseNumberParticleState>()->size(), cells);
    EXPECT_EQ(universe.states().size(), std::size_t { 1 });
}

TEST(Universe, StateStoreEmplacePresenceAndRemoval) {
    Universe universe = make_universe(Float3(0.0f, 0.0f, 0.0f), Float3(2.0f, 2.0f, 2.0f), 1.0f);
    const auto cells = static_cast<std::size_t>(universe.cell_count());

    EXPECT_FALSE(universe.has_state<UniverseNumberParticleState>());

    UniverseNumberParticleState& state = universe.emplace_state<UniverseNumberParticleState>(cells);
    EXPECT_EQ(state.size(), cells);
    EXPECT_TRUE(universe.has_state<UniverseNumberParticleState>());
    EXPECT_NE(universe.state<UniverseNumberParticleState>(), nullptr);

    auto removed = universe.remove_state<UniverseNumberParticleState>();
    ASSERT_TRUE(static_cast<bool>(removed));
    EXPECT_EQ(removed->size(), cells);
    EXPECT_FALSE(universe.has_state<UniverseNumberParticleState>());
    EXPECT_EQ(universe.state<UniverseNumberParticleState>(), nullptr);
}
