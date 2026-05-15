#include "../../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
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
using atlas::system::DsmcSolver;
using atlas::system::SpatialHashingSearcher;
using atlas::universe::UniverseCollisionCountState;
using atlas::universe::UniverseMaxRelativeSpeedState;
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

class DummyDsmcSolver final : public DsmcSolver<float> {
public:
    using DsmcSolver<float>::DsmcSolver;

public:
    void
    apply_collisions(const DsmcSolver<float>::DsmcSolverProbe&, int, float) override { }
};

} // namespace

TEST(DsmcSolver, BaseConstructorCreatesRequiredUniverseStates) {
    // Arrange: create the universe, fluid, and searcher dependencies.
    const auto universe = make_universe();
    const auto fluid    = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    // Act: construct a concrete test implementation of the DSMC base solver.
    const DummyDsmcSolver solver(universe, fluid, searcher, DsmcKernelType::variable_soft_sphere);

    // Assert: construction preserves kernel type and installs required universe states.
    EXPECT_EQ(solver.kernel_type(), DsmcKernelType::variable_soft_sphere);
    ASSERT_TRUE(universe->has_state<UniverseNumberParticleState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseMaxRelativeSpeedState<float>>());
    ASSERT_TRUE(universe->has_state<UniverseCollisionCountState<int>>());
}
