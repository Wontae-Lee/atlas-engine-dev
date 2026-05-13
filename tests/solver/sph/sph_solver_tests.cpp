#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_solver.h>

#include <testkit/testkit.h>

#include <cmath>

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
using atlas::system::SpatialHashingSearcher;
using atlas::system::SphKernelType;
using atlas::system::SphSolver;
using atlas::universe::UniverseFieldForceState;
using atlas::universe::UniverseNumberParticleState;

using FluidPositionState = atlas::fluid::FluidPositionState<float>;
using FluidSpeciesState = atlas::fluid::FluidSpeciesState<float>;
using FluidVelocityState = atlas::fluid::FluidVelocityState<float>;

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.5f)
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
            .with_rest_density(0.1f)
            .with_pressure_coefficient(4.0f)
            .with_dynamic_viscosity(0.05f)
            .with_smoothing_length(0.35f)
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

} // namespace

TEST(SphSolver, ConstructorCreatesRequiredUniverseStates) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct the solver directly.
    const SphSolver<float> solver(
        universe,
        fluid,
        searcher,
        SphKernelType::cubic_spline);

    // Assert: construction preserves kernel type and installs required states.
    EXPECT_EQ(solver.kernel_type(), SphKernelType::cubic_spline);
    ASSERT_TRUE(universe->has_state<UniverseNumberParticleState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseFieldForceState<float>>());
}

TEST(SphSolver, BuilderConstructsSolver) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the solver with a non-default kernel type.
    auto solver = SphSolver<float>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(SphKernelType::wendland_quintic)
                      .build();

    DeviceBuffer<int> allocated_solver;

    // Assert: build preserves configuration and creates required states.
    EXPECT_EQ(solver.kernel_type(), SphKernelType::wendland_quintic);
    EXPECT_TRUE(universe->has_state<UniverseNumberParticleState<float>>());
    EXPECT_TRUE(universe->has_state<UniverseFieldForceState<float>>());
    EXPECT_NO_THROW(solver.solve(&allocated_solver, 0, 0.1f));
}

TEST(SphSolver, SolveIsSafeForEmptyFluid) {
    // Arrange: create a solver attached to an empty fluid.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    SphSolver<float> solver(universe, fluid, searcher);

    // Assert: solving an empty fluid is safe and leaves cell particle count at zero.
    EXPECT_NO_THROW(solver.solve(0.1f));
    EXPECT_EQ(universe->state<UniverseNumberParticleState<float>>()->data()[0], 0.0f);
}

TEST(SphSolver, SolveUpdatesVelocityFromLocalNeighborhood) {
    // Arrange: create a two-particle fluid inside one local neighborhood.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(2);
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.20f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.30f, 0.25f, 0.25f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(0.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(0.0f, 0.0f, 0.0f);
    fluid->state<FluidSpeciesState>()->data()[0] = 0u;
    fluid->state<FluidSpeciesState>()->data()[1] = 0u;

    SphSolver<float> solver(universe, fluid, searcher);

    // Act: solve one SPH step.
    solver.solve(0.01f);

    const auto lhs_velocity = fluid->state<FluidVelocityState>()->data()[0];
    const auto rhs_velocity = fluid->state<FluidVelocityState>()->data()[1];
    const auto cell_force = universe->state<UniverseFieldForceState<float>>()->data()[0];

    // Assert: local pressure interaction pushes particles apart and records cell data.
    EXPECT_LT(lhs_velocity.x, 0.0f);
    EXPECT_GT(rhs_velocity.x, 0.0f);
    EXPECT_GT(universe->state<UniverseNumberParticleState<float>>()->data()[0], 0.0f);
    EXPECT_TRUE(std::isfinite(cell_force.x));
}
