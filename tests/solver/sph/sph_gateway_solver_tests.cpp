#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_gateway_solver.h>

#include <testkit/testkit.h>

namespace {

using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::SearcherHostPtr;
using atlas::tol;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::SpatialHashingSearcher;
using atlas::SphGatewaySolver;
using atlas::SphKernelType;
using atlas::test::vec_near;
using atlas::UniverseNumberParticleState;

using FluidPositionState = atlas::FluidPositionState<float>;
using FluidSpeciesState  = atlas::FluidSpeciesState<float>;
using FluidVelocityState = atlas::FluidVelocityState<float>;

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
            .build());

    HostBuffer<GeneratorHostPtr<float>> generators;
    generators.push_back(nullptr);

    return Fluid<float>::builder()
        .with_buffer_size(8)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

SearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(SphGatewaySolver, BuilderConstructsConfiguredSolver) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the gateway solver with a non-default configuration.
    const auto solver = SphGatewaySolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_kernel_type(SphKernelType::wendland_quintic)
                            .with_group_particle_count(3)
                            .build();

    // Assert: builder configuration is preserved.
    EXPECT_EQ(solver.kernel_type(), SphKernelType::wendland_quintic);
    EXPECT_EQ(solver.group_particle_count(), 3);
}

TEST(SphGatewaySolver, CodecAwareSolveSharesStateWithinEachDeterministicGroup) {
    // Arrange: create a four-particle fluid split into deterministic groups.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(4);
    fluid->state<FluidSpeciesState>()->data()[0] = 0u;
    fluid->state<FluidSpeciesState>()->data()[1] = 0u;
    fluid->state<FluidSpeciesState>()->data()[2] = 0u;
    fluid->state<FluidSpeciesState>()->data()[3] = 0u;

    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.10f, 0.10f, 0.10f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.12f, 0.10f, 0.10f);
    fluid->state<FluidPositionState>()->data()[2] = Vector3F(0.30f, 0.10f, 0.10f);
    fluid->state<FluidPositionState>()->data()[3] = Vector3F(0.32f, 0.10f, 0.10f);

    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(3.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[2] = Vector3F(5.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[3] = Vector3F(7.0f, 0.0f, 0.0f);

    const auto initial_p0 = fluid->state<FluidPositionState>()->data()[0];
    const auto initial_p1 = fluid->state<FluidPositionState>()->data()[1];
    const auto initial_p2 = fluid->state<FluidPositionState>()->data()[2];
    const auto initial_p3 = fluid->state<FluidPositionState>()->data()[3];

    SphGatewaySolver<float> solver(
        universe,
        fluid,
        searcher,
        SphKernelType::standard,
        2);

    DeviceBuffer<int> allocated_solver(static_cast<std::size_t>(universe->number_of_cells()), -1);
    allocated_solver[0] = 0;

    // Act: solve only the selected deterministic cell group.
    solver.solve(&allocated_solver, 0, 0.01f);

    const auto p0 = fluid->state<FluidPositionState>()->data()[0];
    const auto p1 = fluid->state<FluidPositionState>()->data()[1];
    const auto p2 = fluid->state<FluidPositionState>()->data()[2];
    const auto p3 = fluid->state<FluidPositionState>()->data()[3];

    const auto v0 = fluid->state<FluidVelocityState>()->data()[0];
    const auto v1 = fluid->state<FluidVelocityState>()->data()[1];
    const auto v2 = fluid->state<FluidVelocityState>()->data()[2];
    const auto v3 = fluid->state<FluidVelocityState>()->data()[3];

    // Assert: positions remain fixed while velocities are shared within each group.
    EXPECT_TRUE(vec_near(p0, initial_p0, tol));
    EXPECT_TRUE(vec_near(p1, initial_p1, tol));
    EXPECT_TRUE(vec_near(p2, initial_p2, tol));
    EXPECT_TRUE(vec_near(p3, initial_p3, tol));
    EXPECT_TRUE(vec_near(v0, v1, tol));
    EXPECT_TRUE(vec_near(v2, v3, tol));

    EXPECT_GT(universe->state<UniverseNumberParticleState<float>>()->data()[0], 0.0f);
}
