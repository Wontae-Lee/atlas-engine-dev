#include <atlas/universe/universe_view.h>

#include <atlas/math/math.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cstddef>

namespace {

using atlas::Float3;
using atlas::tol;
using atlas::Universe;
using atlas::UniverseAllocatedSolverState;
using atlas::UniverseCollisionCountState;
using atlas::UniverseDsmcView;
using atlas::UniverseMaxRelativeSpeedState;
using atlas::UniverseMaxSigmaGState;
using atlas::UniverseNumberParticleState;

// Universe with 27 cells; each required DSMC state, when added, is sized to it.
Universe
make_universe() {
    return Universe::builder()
        .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Float3(2.0f, 2.0f, 2.0f))
        .with_cell_size(1.0f)
        .build();
}

}

TEST(UniverseDsmcView, DimensionsAgreeWithOwner) {
    Universe universe = make_universe();

    const UniverseDsmcView view = universe.view<UniverseDsmcView>();

    EXPECT_EQ(view.cell_count, universe.cell_count());
    EXPECT_NEAR(view.cell_volume, universe.cell_volume(), tol);
}

TEST(UniverseDsmcView, IncompleteWhenRequiredStatesAbsent) {
    Universe universe = make_universe();

    const UniverseDsmcView view = universe.view<UniverseDsmcView>();

    EXPECT_FALSE(view.is_complete());
    EXPECT_EQ(view.number_particle, nullptr);
    EXPECT_EQ(view.max_relative_speed, nullptr);
    EXPECT_EQ(view.max_sigma_g, nullptr);
    EXPECT_EQ(view.collision_count, nullptr);
    EXPECT_EQ(view.allocated_solver, nullptr);
}

TEST(UniverseDsmcView, CompleteOnceRequiredStatesPresent) {
    Universe universe = make_universe();
    const auto cells = static_cast<std::size_t>(universe.cell_count());

    universe.emplace_state<UniverseNumberParticleState>(cells);
    universe.emplace_state<UniverseMaxRelativeSpeedState>(cells);
    universe.emplace_state<UniverseMaxSigmaGState>(cells);
    universe.emplace_state<UniverseCollisionCountState>(cells);

    const UniverseDsmcView view = universe.view<UniverseDsmcView>();

    EXPECT_TRUE(view.is_complete());
    EXPECT_NE(view.number_particle, nullptr);
    EXPECT_NE(view.max_relative_speed, nullptr);
    EXPECT_NE(view.max_sigma_g, nullptr);
    EXPECT_NE(view.collision_count, nullptr);
    // allocated_solver was never added: a null owning-solver buffer is valid.
    EXPECT_EQ(view.allocated_solver, nullptr);
}

TEST(UniverseDsmcView, AllocatedSolverExcludedFromCompleteness) {
    Universe universe = make_universe();
    const auto cells = static_cast<std::size_t>(universe.cell_count());

    // Only the owning-solver state is present; the mandatory fields are still
    // missing, so is_complete() ignores it and stays false.
    universe.emplace_state<UniverseAllocatedSolverState>(cells);

    const UniverseDsmcView view = universe.view<UniverseDsmcView>();

    EXPECT_NE(view.allocated_solver, nullptr);
    EXPECT_FALSE(view.is_complete());
}
