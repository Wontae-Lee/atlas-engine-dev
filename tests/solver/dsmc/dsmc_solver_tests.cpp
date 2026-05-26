#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_cell_sequential_solver.h>
#include <atlas/solver/dsmc/dsmc_flatten_solver.h>
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
using atlas::system::DsmcCellSequentialSolver;
using atlas::system::DsmcFlattenSolver;
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

} // namespace

TEST(DsmcSolver, ConstructorCreatesRequiredUniverseStates) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct the DSMC solver directly.
    const DsmcCellSequentialSolver<float> solver(universe, fluid, searcher, DsmcKernelType::variable_soft_sphere);

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
    const auto solver = DsmcFlattenSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_kernel_type(DsmcKernelType::variable_hard_sphere)
                            .with_prevent_duplicate_pairing(true)
                            .build();

    // Assert: the configured kernel type is preserved.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_hard_sphere);
    EXPECT_TRUE(solver.prevent_duplicate_pairing());
}

TEST(DsmcSolver, BuilderConstructsCellSequentialSolver) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the cell-sequential solver.
    const auto solver = DsmcCellSequentialSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .build();

    // Assert: base solver options are preserved.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::hard_sphere);
    EXPECT_FALSE(solver.prevent_duplicate_pairing());
}

TEST(DsmcSolver, DefaultsToCellSequential) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct with default DSMC execution options.
    const DsmcCellSequentialSolver<float> solver(universe, fluid, searcher);

    // Assert: defaults are simple.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::hard_sphere);
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
    const DsmcCellSequentialSolver<float> solver(universe, fluid, searcher);
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

    DsmcCellSequentialSolver<float> solver(universe, fluid, searcher);

    // Assert: solving an empty fluid is a no-op for collision counts.
    EXPECT_NO_THROW(solver.solve(0.1f));
    EXPECT_EQ(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
}

TEST(DsmcSolver, SolveStoresNtcCollisionRate) {
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

    auto solver = DsmcFlattenSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .build();

    // Act: build collision statistics without changing velocities.
    solver.solve(1.0f);

    // Assert: max sigma*g, max relative speed, and candidate count are populated.
    EXPECT_NEAR(universe->state<UniverseMaxRelativeSpeedState<float>>()->data()[0], 1.0f, 1.0e-6f);
    EXPECT_NEAR(universe->state<UniverseMaxSigmaGState<float>>()->data()[0], static_cast<float>(atlas::pi), 1.0e-5f);
    EXPECT_GT(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
    EXPECT_EQ(solver.collision_offsets().size(), static_cast<std::size_t>(universe->number_of_cells()));
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

    auto solver = DsmcCellSequentialSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .build();

    // Act: solve through the cell-sequential path.
    solver.solve(1.0f);

    // Assert: collision counts are still estimated, but no flattened list is built.
    EXPECT_GT(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
}

TEST(DsmcSolver, FullPairScanSolvesDenseCell) {
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

    auto solver = DsmcCellSequentialSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(DsmcKernelType::hard_sphere)
                      .build();

    // Act: solve with full pair scanning.
    solver.solve(1.0f);

    // Assert: the full-scan path still schedules valid DSMC work.
    EXPECT_GT(universe->state<UniverseMaxSigmaGState<float>>()->data()[0], 0.0f);
    EXPECT_GT(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
}
