#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

#include <testkit/testkit.h>

namespace {

using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::SpatialHashingSearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::system::DsmcKernelType;
using atlas::system::DsmcApplyMode;
using atlas::system::DsmcMajorantMode;
using atlas::system::DsmcSolver;
using atlas::system::SpatialHashingSearcher;
using atlas::universe::UniverseCollisionCountState;
using atlas::universe::UniverseMaxRelativeSpeedState;
using atlas::universe::UniverseMaxSigmaGState;
using atlas::universe::UniverseNumberParticleState;

using FluidPositionState = atlas::fluid::FluidPositionState<float>;
using FluidSpeciesState  = atlas::fluid::FluidSpeciesState<float>;
using FluidVelocityState = atlas::fluid::FluidVelocityState<float>;

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

FluidHostPtr<float>
make_fluid() {
    HostBuffer<MaterialProperties<float>> properties;
    properties.push_back(
        MaterialProperties<float>::builder()
            .with_type(MaterialType::Molecule)
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .with_reference_diameter(1.0f)
            .build());

    HostBuffer<GeneratorHostPtr<float>> generators;
    generators.push_back(nullptr);

    return Fluid<float>::builder()
        .with_buffer_size(4)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

SpatialHashingSearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

void
populate_four_particle_cell(const FluidHostPtr<float>& fluid) {
    fluid->set_particle_count(4);
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.10f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.30f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[2] = Vector3F(0.50f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[3] = Vector3F(0.70f, 0.25f, 0.25f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(0.0f, 1.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[2] = Vector3F(0.0f, 0.0f, 1.0f);
    fluid->state<FluidVelocityState>()->data()[3] = Vector3F(-1.0f, 0.0f, 0.0f);
    fluid->state<FluidSpeciesState>()->data()[0]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[1]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[2]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[3]  = 0u;
}

float
velocity_difference_squared(const FluidHostPtr<float>& lhs,
                            const FluidHostPtr<float>& rhs) {
    float delta = 0.0f;
    for (std::size_t i = 0; i < lhs->particle_count(); ++i) {
        delta += (lhs->state<FluidVelocityState>()->data()[i] - rhs->state<FluidVelocityState>()->data()[i]).length_squared();
    }
    return delta;
}

} // namespace

TEST(DsmcSolver, ConstructorCreatesRequiredUniverseStates) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct the DSMC solver directly.
    const DsmcSolver<float> solver(universe, fluid, searcher, DsmcKernelType::variable_soft_sphere);

    // Assert: construction preserves kernel type and installs required universe states.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_soft_sphere);
    ASSERT_TRUE(universe->has_state<UniverseNumberParticleState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseMaxRelativeSpeedState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseMaxSigmaGState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseCollisionCountState<int>>());
}

TEST(DsmcSolver, BuilderConstructsSolverWithKernelType) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the solver with a non-default kernel type.
    const auto solver = DsmcSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_kernel_type(DsmcKernelType::variable_hard_sphere)
                            .build();

    // Assert: the configured kernel type is preserved.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_hard_sphere);
}

TEST(DsmcSolver, BuilderConstructsSolverWithApplyMode) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the solver with the lock-free cell-sequential apply path.
    const auto solver = DsmcSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_apply_mode(DsmcApplyMode::cell_sequential)
                            .build();

    // Assert: the configured apply mode is preserved.
    EXPECT_EQ(solver.apply_mode(), DsmcApplyMode::cell_sequential);
}

TEST(DsmcSolver, BuilderConstructsSolverWithMajorantOptions) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the solver with exact majorant estimation.
    const auto solver = DsmcSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_majorant_mode(DsmcMajorantMode::exact_all_pairs)
                            .with_majorant_sample_count(8)
                            .with_majorant_safety_factor(1.5f)
                            .with_majorant_decay_factor(0.9f)
                            .with_pre_collision_snapshot(true)
                            .with_diagnostics_enabled(true)
                            .build();

    // Assert: the configured majorant controls are preserved.
    EXPECT_EQ(solver.majorant_mode(), DsmcMajorantMode::exact_all_pairs);
    EXPECT_EQ(solver.majorant_sample_count(), 8);
    EXPECT_FLOAT_EQ(solver.majorant_safety_factor(), 1.5f);
    EXPECT_FLOAT_EQ(solver.majorant_decay_factor(), 0.9f);
    EXPECT_TRUE(solver.use_pre_collision_snapshot());
    EXPECT_TRUE(solver.diagnostics_enabled());
}

TEST(DsmcSolver, DefaultsToCellSequentialWithoutSnapshotPrecheck) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct with default DSMC execution options.
    const DsmcSolver<float> solver(universe, fluid, searcher);

    // Assert: the default path is the lock-free cell-sequential solver.
    EXPECT_EQ(solver.apply_mode(), DsmcApplyMode::cell_sequential);
    EXPECT_EQ(solver.majorant_mode(), DsmcMajorantMode::exact_all_pairs);
    EXPECT_FLOAT_EQ(solver.majorant_safety_factor(), 1.0f);
    EXPECT_FLOAT_EQ(solver.majorant_decay_factor(), 1.0f);
    EXPECT_FALSE(solver.use_pre_collision_snapshot());
    EXPECT_FALSE(solver.diagnostics_enabled());
}

TEST(DsmcSolver, ConstructorResizesExistingUniverseStates) {
    // Arrange: install undersized DSMC states before constructing the solver.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    universe->emplace_state<UniverseNumberParticleState<float>>(0);
    universe->emplace_state<UniverseMaxRelativeSpeedState<float>>(0);
    universe->emplace_state<UniverseMaxSigmaGState<float>>(0);
    universe->emplace_state<UniverseCollisionCountState<int>>(0);

    // Act: construct the solver, which should normalize state sizes.
    const DsmcSolver<float> solver(universe, fluid, searcher);
    (void)solver;

    // Assert: all DSMC cell states match the universe cell count.
    const auto cells = static_cast<std::size_t>(universe->number_of_cells());
    EXPECT_EQ(universe->state<UniverseNumberParticleState<float>>()->data().size(), cells);
    EXPECT_EQ(universe->state<UniverseMaxRelativeSpeedState<float>>()->data().size(), cells);
    EXPECT_EQ(universe->state<UniverseMaxSigmaGState<float>>()->data().size(), cells);
    EXPECT_EQ(universe->state<UniverseCollisionCountState<int>>()->data().size(), cells);
}

TEST(DsmcSolver, SolveIsSafeForEmptyFluid) {
    // Arrange: create a solver attached to an empty fluid.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DsmcSolver<float> solver(universe, fluid, searcher);

    // Assert: solving an empty fluid is a no-op for collision counts.
    EXPECT_NO_THROW(solver.solve(0.1f));
    EXPECT_EQ(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
}

TEST(DsmcSolver, SolveStoresNtcCollisionRateMajorant) {
    // Arrange: put two active hard-sphere particles in the single universe cell.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(2);
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.25f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.75f, 0.25f, 0.25f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(0.0f, 0.0f, 0.0f);
    fluid->state<FluidSpeciesState>()->data()[0]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[1]  = 0u;

    auto solver = DsmcSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .with_apply_mode(DsmcApplyMode::flattened_atomic)
                      .with_diagnostics_enabled(true)
                      .build();

    // Act: build collision statistics without changing velocities.
    solver.solve(1.0f);

    // Assert: max sigma*g, max relative speed, and candidate count are populated.
    EXPECT_NEAR(universe->state<UniverseMaxRelativeSpeedState<float>>()->data()[0], 1.0f, 1.0e-6f);
    EXPECT_NEAR(universe->state<UniverseMaxSigmaGState<float>>()->data()[0], static_cast<float>(atlas::pi), 1.0e-5f);
    EXPECT_GT(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
    EXPECT_EQ(
        solver.flattened_collision_cells().size(),
        static_cast<std::size_t>(universe->state<UniverseCollisionCountState<int>>()->data()[0]));
    EXPECT_EQ(solver.majorant_violation_count(), 0);
    EXPECT_GT(solver.accepted_collision_count(), 0);
}

TEST(DsmcSolver, CellSequentialApplyDoesNotBuildFlattenedWorkload) {
    // Arrange: put two active hard-sphere particles in the single universe cell.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(2);
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.25f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.75f, 0.25f, 0.25f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(0.0f, 0.0f, 0.0f);
    fluid->state<FluidSpeciesState>()->data()[0]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[1]  = 0u;

    auto solver = DsmcSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .with_apply_mode(DsmcApplyMode::cell_sequential)
                      .build();

    // Act: solve through the cell-sequential path.
    solver.solve(1.0f);

    // Assert: collision counts are still estimated, but no flattened list is built.
    EXPECT_GT(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
    EXPECT_TRUE(solver.flattened_collision_cells().empty());
}

TEST(DsmcSolver, SampledAdaptiveMajorantSolvesDenseCellWithoutExactPairBudget) {
    // Arrange: put four active hard-sphere particles in the single universe cell.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(4);
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.10f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.30f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[2] = Vector3F(0.50f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[3] = Vector3F(0.70f, 0.25f, 0.25f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(0.0f, 1.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[2] = Vector3F(0.0f, 0.0f, 1.0f);
    fluid->state<FluidVelocityState>()->data()[3] = Vector3F(-1.0f, 0.0f, 0.0f);
    fluid->state<FluidSpeciesState>()->data()[0]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[1]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[2]  = 0u;
    fluid->state<FluidSpeciesState>()->data()[3]  = 0u;

    auto solver = DsmcSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .with_majorant_mode(DsmcMajorantMode::sampled_adaptive)
                      .with_majorant_sample_count(1)
                      .with_majorant_safety_factor(1.25f)
                      .build();

    // Act: solve with a dense-cell sampled majorant budget lower than all pairs.
    solver.solve(1.0f);

    // Assert: the sampled/adaptive path still schedules valid DSMC work.
    EXPECT_GT(universe->state<UniverseMaxSigmaGState<float>>()->data()[0], 0.0f);
    EXPECT_GT(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
}

TEST(DsmcSolver, DiagnosticsAreDisabledByDefault) {
    // Arrange: put active particles in the single universe cell.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);
    populate_four_particle_cell(fluid);

    DsmcSolver<float> solver(universe, fluid, searcher);

    // Act: solve with default diagnostics disabled.
    solver.solve(1.0f);

    // Assert: diagnostic counters stay unavailable/zero unless explicitly enabled.
    EXPECT_FALSE(solver.diagnostics_enabled());
    EXPECT_EQ(solver.majorant_violation_count(), 0);
    EXPECT_EQ(solver.accepted_collision_count(), 0);
}

TEST(DsmcSolver, SnapshotPrecheckDoesNotIncreaseAcceptedCount) {
    // Arrange: create two identical flattened-atomic simulations.
    const auto universe_without_snapshot = make_universe();
    const auto fluid_without_snapshot    = make_fluid();
    const auto searcher_without_snapshot = make_searcher(universe_without_snapshot, fluid_without_snapshot);
    populate_four_particle_cell(fluid_without_snapshot);

    const auto universe_with_snapshot = make_universe();
    const auto fluid_with_snapshot    = make_fluid();
    const auto searcher_with_snapshot = make_searcher(universe_with_snapshot, fluid_with_snapshot);
    populate_four_particle_cell(fluid_with_snapshot);

    auto solver_without_snapshot = DsmcSolver<float>::builder()
                                       .with_universe(universe_without_snapshot)
                                       .with_fluid(fluid_without_snapshot)
                                       .with_searcher(searcher_without_snapshot)
                                       .with_apply_mode(DsmcApplyMode::flattened_atomic)
                                       .with_majorant_mode(DsmcMajorantMode::exact_all_pairs)
                                       .with_diagnostics_enabled(true)
                                       .build();

    auto solver_with_snapshot = DsmcSolver<float>::builder()
                                    .with_universe(universe_with_snapshot)
                                    .with_fluid(fluid_with_snapshot)
                                    .with_searcher(searcher_with_snapshot)
                                    .with_apply_mode(DsmcApplyMode::flattened_atomic)
                                    .with_majorant_mode(DsmcMajorantMode::exact_all_pairs)
                                    .with_pre_collision_snapshot(true)
                                    .with_diagnostics_enabled(true)
                                    .build();

    // Act: solve both systems with identical initial state.
    solver_without_snapshot.solve(1.0f);
    solver_with_snapshot.solve(1.0f);

    // Assert: snapshot precheck is an extra reject gate and should not increase accepts.
    EXPECT_LE(solver_with_snapshot.accepted_collision_count(), solver_without_snapshot.accepted_collision_count());
    EXPECT_GE(velocity_difference_squared(fluid_without_snapshot, fluid_with_snapshot), 0.0f);
}

TEST(DsmcSolver, SampledMajorantReportsNoViolationWhenBootstrappedFromExactState) {
    // Arrange: use sampled mode with one sample, but start from the exact bootstrap.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);
    populate_four_particle_cell(fluid);

    auto solver = DsmcSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_majorant_mode(DsmcMajorantMode::sampled)
                      .with_majorant_sample_count(1)
                      .with_majorant_safety_factor(1.05f)
                      .with_majorant_decay_factor(0.99f)
                      .with_diagnostics_enabled(true)
                      .build();

    // Act: first solve bootstraps exact majorant because previous max is zero.
    solver.solve(1.0e-9f);
    solver.solve(1.0e-9f);

    // Assert: unchanged velocities and decayed exact carry-over should avoid violations.
    EXPECT_EQ(solver.majorant_violation_count(), 0);
}
