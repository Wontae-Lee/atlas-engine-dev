#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_disjoint_pair_solver.h>

#include <testkit/testkit.h>

namespace {

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
using atlas::system::DsmcDisjointPairSolver;
using atlas::system::DsmcKernelType;
using atlas::system::SpatialHashingSearcher;
using atlas::universe::UniverseCollisionCountState;
using atlas::universe::UniverseNumberParticleState;

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
            .with_collision_diameter(1.0f)
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

TEST(DsmcDisjointPairSolver, ConstructorCreatesRequiredUniverseStates) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct the solver directly.
    const DsmcDisjointPairSolver<float> solver(universe, fluid, searcher);

    // Assert: construction installs required universe states.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::hard_sphere);
    ASSERT_TRUE(universe->has_state<UniverseNumberParticleState<float>>());
}

TEST(DsmcDisjointPairSolver, BuilderConstructsSolverWithKernelType) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: build the solver with a non-default kernel type.
    const auto solver = DsmcDisjointPairSolver<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_kernel_type(DsmcKernelType::variable_soft_sphere)
                            .build();

    // Assert: the configured kernel type is preserved.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_soft_sphere);
}

TEST(DsmcDisjointPairSolver, SolveIsSafeForEmptyFluid) {
    // Arrange: create a solver attached to an empty fluid.
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DsmcDisjointPairSolver<float> solver(universe, fluid, searcher);

    // Assert: solving an empty fluid is a no-op for collision counts.
    EXPECT_NO_THROW(solver.solve(0.1f));
    EXPECT_EQ(universe->state<UniverseCollisionCountState<int>>()->data()[0], 0);
}
