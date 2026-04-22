#include "../../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_solver.h>

#include <testkit/testkit.h>

#include <cmath>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(Vec3(0, 0, 0))
        .with_upper_corner(Vec3(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::MatrialProperties<T>> properties;
    properties.push_back(
        atlas::MatrialProperties<T>::builder()
            .with_type(atlas::MaterialType::Molecule)
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .with_rest_density(0.1f)
            .with_pressure_coefficient(4.0f)
            .with_dynamic_viscosity(0.05f)
            .with_smoothing_length(0.35f)
            .build());

    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators;
    generators.push_back(nullptr);

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(4)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::SpatialHashingSearcherHostPtr<T>
make_searcher(const atlas::UniverseHostPtr<T>& universe,
              const atlas::FluidHostPtr<T>& fluid) {
    return atlas::system::SpatialHashingSearcher<T>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(SphSolver, ConstructorCreatesRequiredUniverseStates) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const atlas::system::SphSolver<T> solver(
        universe,
        fluid,
        searcher,
        atlas::system::SphKernelType::cubic_spline);

    EXPECT_EQ(solver.kernel_type(), atlas::system::SphKernelType::cubic_spline);
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseNumberParticleState<T>>());
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseFieldForceState<T>>());
}

TEST(SphSolver, BuilderConstructsSolver) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    auto solver = atlas::system::SphSolver<T>::builder()
                      .with_universe(universe)
                      .with_fluid(fluid)
                      .with_searcher(searcher)
                      .with_kernel_type(atlas::system::SphKernelType::wendland_quintic)
                      .build();

    atlas::DeviceBuffer<int> allocated_solver;

    EXPECT_EQ(solver.kernel_type(), atlas::system::SphKernelType::wendland_quintic);
    EXPECT_TRUE(universe->has_state<atlas::universe::UniverseNumberParticleState<T>>());
    EXPECT_TRUE(universe->has_state<atlas::universe::UniverseFieldForceState<T>>());
    EXPECT_NO_THROW(solver.solve(&allocated_solver, 0, 0.1f));
}

TEST(SphSolver, SolveIsSafeForEmptyFluid) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    atlas::system::SphSolver<T> solver(universe, fluid, searcher);

    EXPECT_NO_THROW(solver.solve(0.1f));
    EXPECT_EQ(universe->state<atlas::universe::UniverseNumberParticleState<T>>()->data()[0], 0.0f);
}

TEST(SphSolver, SolveUpdatesVelocityFromLocalNeighborhood) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(2);
    fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[0] = Vec3(0.20f, 0.25f, 0.25f);
    fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[1] = Vec3(0.30f, 0.25f, 0.25f);
    fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[0] = Vec3(0.0f, 0.0f, 0.0f);
    fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[1] = Vec3(0.0f, 0.0f, 0.0f);
    fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data()[0] = 0u;
    fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data()[1] = 0u;

    atlas::system::SphSolver<T> solver(universe, fluid, searcher);

    solver.solve(0.01f);

    const auto lhs_velocity = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[0];
    const auto rhs_velocity = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[1];
    const auto cell_force = universe->state<atlas::universe::UniverseFieldForceState<T>>()->data()[0];

    EXPECT_LT(lhs_velocity.x, 0.0f);
    EXPECT_GT(rhs_velocity.x, 0.0f);
    EXPECT_GT(universe->state<atlas::universe::UniverseNumberParticleState<T>>()->data()[0], 0.0f);
    EXPECT_TRUE(std::isfinite(cell_force.x));
}
