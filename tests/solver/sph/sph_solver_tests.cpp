#include <atlas/solver/sph/sph_solver.h>

#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generate.h>
#include <atlas/material/material_properties.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::SphKernelType;
using atlas::SphSolver;
using atlas::Universe;
using atlas::UniverseFieldForceState;
using atlas::UniverseHostPtr;
using atlas::UniverseNumberParticleState;
using atlas::Float3;

UniverseHostPtr
make_universe() {
    return Universe::builder()
        .with_lower_corner(Float3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Float3(1.0f, 1.0f, 1.0f))
        .with_cell_size(0.5f)
        .make_host_shared();
}

FluidHostPtr
make_fluid() {
    HostBuffer<MaterialProperties> properties;
    properties.push_back(
        MaterialProperties::builder()
            .with_type(MaterialType::molecule)
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .with_rest_density(0.1f)
            .with_pressure_coefficient(4.0f)
            .with_dynamic_viscosity(0.05f)
            .build());

    HostBuffer<GeneratorHostPtr> generators;
    generators.push_back(nullptr);

    return Fluid::builder()
        .with_buffer_size(4)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

SearcherHostPtr
make_searcher(const UniverseHostPtr& universe,
              const FluidHostPtr& fluid) {
    return SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

}

TEST(SphSolver, ConstructorCreatesRequiredUniverseStates) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const SphSolver solver(
        universe,
        fluid,
        searcher,
        SphKernelType::cubic_spline);

    EXPECT_EQ(solver.kernel_type(), SphKernelType::cubic_spline);
    ASSERT_TRUE(universe->has_state<UniverseNumberParticleState>());
    ASSERT_TRUE(universe->has_state<UniverseFieldForceState>());
}

TEST(SphSolver, BuilderConstructsSolver) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto solver = SphSolver::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(SphKernelType::wendland_quintic)
                      .build();

    DeviceBuffer<int> allocated_solver;

    EXPECT_EQ(solver.kernel_type(), SphKernelType::wendland_quintic);
    EXPECT_TRUE(universe->has_state<UniverseNumberParticleState>());
    EXPECT_TRUE(universe->has_state<UniverseFieldForceState>());
    EXPECT_NO_THROW(solver.solve(&allocated_solver, 0, 0.1f));
}

TEST(SphSolver, SolveIsSafeForEmptyFluid) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    SphSolver solver(universe, fluid, searcher);

    EXPECT_NO_THROW(solver.solve(0.1f));
    EXPECT_EQ(universe->state<UniverseNumberParticleState>()->data()[0], 0.0f);
}

TEST(SphSolver, SolveUpdatesVelocityFromLocalNeighborhood) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(2);
    fluid->state<FluidPositionState>()->data()[0] = Float3(0.20f, 0.25f, 0.25f);
    fluid->state<FluidPositionState>()->data()[1] = Float3(0.30f, 0.25f, 0.25f);
    fluid->state<FluidVelocityState>()->data()[0] = Float3(0.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Float3(0.0f, 0.0f, 0.0f);
    fluid->state<FluidSpeciesState>()->data()[0] = 0u;
    fluid->state<FluidSpeciesState>()->data()[1] = 0u;

    SphSolver solver(universe, fluid, searcher);

    solver.solve(0.01f);

    const Float3 lhs_velocity = fluid->state<FluidVelocityState>()->data()[0];
    const Float3 rhs_velocity = fluid->state<FluidVelocityState>()->data()[1];
    const Float3 cell_force = universe->state<UniverseFieldForceState>()->data()[0];

    EXPECT_LT(lhs_velocity.x, 0.0f);
    EXPECT_GT(rhs_velocity.x, 0.0f);
    EXPECT_GT(universe->state<UniverseNumberParticleState>()->data()[0], 0.0f);
    EXPECT_TRUE(std::isfinite(cell_force.x));
}
