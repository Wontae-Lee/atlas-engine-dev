#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/hybrid/hybrid_dsmc_sph_solver.h>

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
using atlas::DsmcKernelType;
using atlas::HybridDsmcSphSolver;
using atlas::SpatialHashingSearcher;
using atlas::SphKernelType;
using atlas::test::vec_near;
using atlas::UniverseCollisionCountState;
using atlas::UniverseFieldForceState;
using atlas::UniverseMaxRelativeSpeedState;
using atlas::UniverseMaxSigmaGState;
using atlas::UniverseNumberParticleState;

using FluidPositionState = atlas::FluidPositionState<float>;
using FluidSpeciesState  = atlas::FluidSpeciesState<float>;
using FluidVelocityState = atlas::FluidVelocityState<float>;

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(0.25f)
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
            .with_rest_density(0.1f)
            .with_pressure_coefficient(4.0f)
            .with_dynamic_viscosity(0.05f)
            .build());

    HostBuffer<GeneratorHostPtr<float>> generators;
    generators.push_back(nullptr);

    return Fluid<float>::builder()
        .with_buffer_size(4)
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

void
populate_adjacent_cell_pair(const FluidHostPtr<float>& fluid) {
    fluid->set_particle_count(2);
    fluid->state<FluidSpeciesState>()->data()[0] = 0u;
    fluid->state<FluidSpeciesState>()->data()[1] = 0u;
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.24f, 0.10f, 0.10f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.26f, 0.10f, 0.10f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(3.0f, 0.0f, 0.0f);
}

void
populate_far_pair(const FluidHostPtr<float>& fluid) {
    fluid->set_particle_count(2);
    fluid->state<FluidSpeciesState>()->data()[0] = 0u;
    fluid->state<FluidSpeciesState>()->data()[1] = 0u;
    fluid->state<FluidPositionState>()->data()[0] = Vector3F(0.10f, 0.10f, 0.10f);
    fluid->state<FluidPositionState>()->data()[1] = Vector3F(0.90f, 0.10f, 0.10f);
    fluid->state<FluidVelocityState>()->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    fluid->state<FluidVelocityState>()->data()[1] = Vector3F(-1.0f, 0.0f, 0.0f);
}

} // namespace

TEST(HybridDsmcSphSolver, BuilderConstructsConfiguredSolverAndStates) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build a configured hybrid solver.
    const auto solver = HybridDsmcSphSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_grouping_length(0.08f)
                            .with_sph_particle_threshold(2)
                            .with_sph_kernel_type(SphKernelType::wendland_quintic)
                            .with_dsmc_kernel_type(DsmcKernelType::hard_sphere)
                            .with_pairing_without_replacement(true)
                            .build();

    // Assert: configuration and required universe states are installed.
    EXPECT_EQ(solver.grouping_length(), 0.08f);
    EXPECT_EQ(solver.sph_particle_threshold(), 2);
    EXPECT_EQ(solver.sph_kernel_type(), SphKernelType::wendland_quintic);
    EXPECT_EQ(solver.dsmc_kernel_type(), DsmcKernelType::hard_sphere);
    EXPECT_TRUE(solver.pairing_without_replacement());
    EXPECT_TRUE(universe->has_state<UniverseNumberParticleState<float>>());
    EXPECT_TRUE(universe->has_state<UniverseFieldForceState<float>>());
    EXPECT_TRUE(universe->has_state<UniverseMaxRelativeSpeedState<float>>());
    EXPECT_TRUE(universe->has_state<UniverseMaxSigmaGState<float>>());
    EXPECT_TRUE(universe->has_state<UniverseCollisionCountState<int>>());
}

TEST(HybridDsmcSphSolver, GroupsSphParticlesAcrossAdjacentCells) {
    // Arrange: place two close particles on opposite sides of a search-cell boundary.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);
    populate_adjacent_cell_pair(fluid);

    HybridDsmcSphSolver<float> solver(
        universe,
        fluid,
        searcher,
        0.08f,
        2,
        SphKernelType::standard,
        DsmcKernelType::hard_sphere);

    // Act: solve one hybrid step.
    solver.solve(0.01f);

    const auto v0 = fluid->state<FluidVelocityState>()->data()[0];
    const auto v1 = fluid->state<FluidVelocityState>()->data()[1];

    // Assert: cross-cell grouping treats both particles as one SPH group.
    EXPECT_TRUE(vec_near(v0, v1, tol));
}

TEST(HybridDsmcSphSolver, CodecAllocationIsIgnored) {
    // Arrange: provide an allocation buffer that would exclude every cell in a cell-filtered solver.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);
    populate_adjacent_cell_pair(fluid);

    HybridDsmcSphSolver<float> solver(
        universe,
        fluid,
        searcher,
        0.08f,
        2,
        SphKernelType::standard,
        DsmcKernelType::hard_sphere);

    DeviceBuffer<int> allocated_solver(static_cast<std::size_t>(universe->number_of_cells()), -1);

    // Act: the codec-aware overload should still run over the whole domain.
    solver.solve(&allocated_solver, 7, 0.01f);

    const auto v0 = fluid->state<FluidVelocityState>()->data()[0];
    const auto v1 = fluid->state<FluidVelocityState>()->data()[1];

    // Assert: allocation does not prevent global length-based SPH grouping.
    EXPECT_TRUE(vec_near(v0, v1, tol));
}

TEST(HybridDsmcSphSolver, DsmcParticlesAreGroupedByLengthBeforeCollision) {
    // Arrange: two DSMC particles are outside each other's grouping-length neighborhood.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);
    populate_far_pair(fluid);

    const auto initial_v0 = fluid->state<FluidVelocityState>()->data()[0];
    const auto initial_v1 = fluid->state<FluidVelocityState>()->data()[1];

    HybridDsmcSphSolver<float> solver(
        universe,
        fluid,
        searcher,
        0.08f,
        3,
        SphKernelType::standard,
        DsmcKernelType::hard_sphere);

    // Act: if DSMC used one global pool, this large dt would make the pair eligible.
    solver.solve(10.0f);

    const auto v0 = fluid->state<FluidVelocityState>()->data()[0];
    const auto v1 = fluid->state<FluidVelocityState>()->data()[1];

    // Assert: separated DSMC groups do not collide with each other.
    EXPECT_TRUE(vec_near(v0, initial_v0, tol));
    EXPECT_TRUE(vec_near(v1, initial_v1, tol));
}
