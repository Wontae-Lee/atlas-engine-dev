#include "dsmc_test_utils.h"

#include <atlas/fluid/fluid_state.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/universe/universe_state.h>

#include <testkit/testkit.h>

namespace {

class ExposedDsmcSolver final : public atlas::DsmcSolver<float> {
public:
    using atlas::DsmcSolver<float>::DsmcSolver;

    const Probe& probe() const noexcept {
        return _probe;
    }
};

ExposedDsmcSolver
make_solver() {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);
    return ExposedDsmcSolver(universe, fluid, searcher);
}

} // namespace

TEST(DsmcSolver, ConstructorStoresKernelAndWorkloadTypes) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);

    ExposedDsmcSolver solver(
        universe,
        fluid,
        searcher,
        atlas::DsmcKernelType::variable_hard_sphere,
        atlas::DsmcCollisionWorkloadType::flatten);

    EXPECT_EQ(solver.kernel_type(), atlas::DsmcKernelType::variable_hard_sphere);
    EXPECT_EQ(solver.workload_type(), atlas::DsmcCollisionWorkloadType::flatten);
}

TEST(DsmcSolver, SetWorkloadTypeChangesCollisionSchedulingMode) {
    auto solver = make_solver();

    solver.set_workload_type(atlas::DsmcCollisionWorkloadType::flatten);

    EXPECT_EQ(solver.workload_type(), atlas::DsmcCollisionWorkloadType::flatten);
}

TEST(DsmcSolver, EnsureStatesCreatesAllDsmcUniverseStates) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);
    ExposedDsmcSolver solver(universe, fluid, searcher);

    solver.ensure_states();

    const auto expected_size = static_cast<std::size_t>(universe->number_of_cells());
    ASSERT_NE(universe->state<atlas::UniverseNumberParticleState<float>>(), nullptr);
    ASSERT_NE(universe->state<atlas::UniverseMaxRelativeSpeedState<float>>(), nullptr);
    ASSERT_NE(universe->state<atlas::UniverseMaxSigmaGState<float>>(), nullptr);
    ASSERT_NE(universe->state<atlas::UniverseCollisionRemainderState<float>>(), nullptr);
    ASSERT_NE(universe->state<atlas::UniverseCollisionCountState<int>>(), nullptr);
    EXPECT_EQ(universe->state<atlas::UniverseNumberParticleState<float>>()->data().size(), expected_size);
    EXPECT_EQ(universe->state<atlas::UniverseCollisionCountState<int>>()->data().size(), expected_size);
}

TEST(DsmcSolver, ResetStatesClearsCollisionStateBuffers) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);
    ExposedDsmcSolver solver(universe, fluid, searcher);
    solver.ensure_states();

    universe->state<atlas::UniverseNumberParticleState<float>>()->data()[0] = 2.0f;
    universe->state<atlas::UniverseMaxRelativeSpeedState<float>>()->data()[0] = 3.0f;
    universe->state<atlas::UniverseMaxSigmaGState<float>>()->data()[0] = 4.0f;
    universe->state<atlas::UniverseCollisionRemainderState<float>>()->data()[0] = 0.5f;
    universe->state<atlas::UniverseCollisionCountState<int>>()->data()[0] = 6;

    solver.reset_states();

    EXPECT_EQ(universe->state<atlas::UniverseNumberParticleState<float>>()->data()[0], 0.0f);
    EXPECT_EQ(universe->state<atlas::UniverseMaxRelativeSpeedState<float>>()->data()[0], 0.0f);
    EXPECT_EQ(universe->state<atlas::UniverseMaxSigmaGState<float>>()->data()[0], 0.0f);
    EXPECT_EQ(universe->state<atlas::UniverseCollisionRemainderState<float>>()->data()[0], 0.0f);
    EXPECT_EQ(universe->state<atlas::UniverseCollisionCountState<int>>()->data()[0], 0);
}

TEST(DsmcSolver, MakeProbeCapturesFluidUniverseAndSearcherViews) {
    const auto universe = atlas::test::make_dsmc_universe();
    const auto fluid = atlas::test::make_dsmc_fluid();
    const auto searcher = atlas::test::make_dsmc_searcher(universe, fluid);
    ExposedDsmcSolver solver(universe, fluid, searcher);

    fluid->set_particle_count(1);
    fluid->state<atlas::FluidPositionState<float>>()->data()[0] = atlas::Vector3F(0.0f, 0.0f, 0.0f);
    fluid->state<atlas::FluidVelocityState<float>>()->data()[0] = atlas::Vector3F(1.0f, 0.0f, 0.0f);
    searcher->build();
    solver.make_probe();

    const auto& probe = solver.probe();
    EXPECT_NE(probe.velocity_ptr, nullptr);
    EXPECT_EQ(probe.internal_energy_ptr, nullptr);
    EXPECT_NE(probe.species_ptr, nullptr);
    EXPECT_NE(probe.properties_ptr, nullptr);
    EXPECT_NE(probe.number_particle_ptr, nullptr);
    EXPECT_NE(probe.max_relative_speed_ptr, nullptr);
    EXPECT_NE(probe.max_sigma_g_ptr, nullptr);
    EXPECT_NE(probe.collision_remainder_ptr, nullptr);
    EXPECT_NE(probe.collision_count_ptr, nullptr);
    EXPECT_EQ(probe.particle_count, 1);
    EXPECT_EQ(probe.species_count, 1);
    EXPECT_EQ(probe.num_of_cells, universe->number_of_cells());
    EXPECT_EQ(probe.cell_volume, universe->cell_volume());
    EXPECT_EQ(probe.statistical_weight, fluid->statistical_weight());
}

TEST(DsmcSolver, MeasureCollisionStatisticsAcceptsEmptyProbe) {
    ExposedDsmcSolver solver;

    EXPECT_TRUE(solver.measure_collision_statistics(nullptr, 0, 0.0f));
}

TEST(DsmcSolver, ApplyCollisionAcceptsEmptyProbe) {
    ExposedDsmcSolver solver;

    EXPECT_NO_THROW(solver.apply_collision(nullptr, 0, 0.0f));
}

TEST(DsmcSolver, ParticleAtMapsLocalCellIndexToParticleIndex) {
    const int indices[] = { 3, 1, 2 };

    EXPECT_EQ(atlas::DsmcSolver<float>::particle_at(1, 0, 3, 4, indices), 1);
    EXPECT_EQ(atlas::DsmcSolver<float>::particle_at(-1, 0, 3, 4, indices), -1);
    EXPECT_EQ(atlas::DsmcSolver<float>::particle_at(3, 0, 3, 4, indices), -1);
    EXPECT_EQ(atlas::DsmcSolver<float>::particle_at(0, 0, 3, 2, indices), -1);
}

TEST(DsmcSolver, CollidePairRejectsInvalidParticleIndices) {
    atlas::DsmcProbe<float> probe;
    const int indices[] = { 0 };
    probe.indices_ptr = indices;
    probe.particle_count = 1;

    EXPECT_FALSE(atlas::DsmcSolver<float>::collide_pair(
        probe,
        0,
        0,
        0,
        1,
        -1,
        0,
        1.0f));
}
